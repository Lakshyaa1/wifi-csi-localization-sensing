"""
wimans_transfer.py - WiMANS cut down to look like our ESP32 setup.

    python wimans_transfer.py <wimans root> <our dataset.npz>

WiMANS clips used: 2.4 GHz, 0 or 1 person, all 3 rooms (classroom, meeting_room, empty_room).
Each clip (about 3000 packets x 3 tx x 3 rx x 30 subcarriers, 3 s) is resampled to 100 Hz (mean of 10
packets) and 30 -> 52 subcarriers by linear interpolation, then goes through the same prepare() as our data
(relative change per subcarrier, 50 Hz). "Cut" input = 2 antenna pairs of one transmitter, like our 2 boards.

Part A - hardware or physics? (WiMANS only, 4 classes like ours: empty = 0 people, sit_down, walk, wave)
  learn in the classroom; test within the classroom (5-fold) and in the other two rooms with source only,
  AdaBN and SHOT-IM (adapt.py settings, adapted on all target clips, labels used only for scoring).
  Two inputs: all 9 antenna pairs, and cut to 2 pairs.
Part B - pretrained backbone for our data (the McQueen recipe)
  pretrain SmallCNN on all WiMANS clips (10 classes: 9 activities + empty, 3 rooms, cut input; each clip
  gives 3 samples, one per transmitter), then on our clips:
    scratch:          plain CNN as in train_cnn.py
    frozen:           pretrained body fixed, new final layer learned on our clips
    frozen + our BN:  same, but batch-norm averages re-measured on our training clips first
    fine-tuned:       pretrained body + new final layer, all learned (body lr 1e-4, final layer 1e-3)
  tested with leave-one-session-out and sessions 1-3 -> 4, 5 seeds.
All settings fixed before running; last-epoch scores.
"""
import copy, os, sys, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from train_cnn import SmallCNN, prepare, fit, predict, EPOCHS, BATCH
from adapt import adabn, shot_im, predict_proba

SEEDS = 5
ACTS = ["nothing", "walk", "rotation", "jump", "wave", "lie_down", "pick_up", "sit_down", "stand_up"]
CUT = [(0, 0), (0, 2)]           # (tx, rx) pairs used as our "2 boards"
PRE_EPOCHS, PRE_BATCH, PRE_LR = 30, 64, 1e-3
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_wimans(root, cache="wimans_24_01.npz"):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["act"], z["env"]
    d = pd.read_csv(f"{root}/annotation.csv", dtype=str)
    d = d[(d.wifi_band == "2.4") & d.number_of_users.isin(["0", "1"])].reset_index(drop=True)
    acts = d[[f"user_{k}_activity" for k in range(1, 7)]].bfill(axis=1).iloc[:, 0].fillna("empty")
    X = np.zeros((len(d), 3, 3, 52, 300), np.float32)
    sc = np.linspace(0, 29, 52)
    lo, w = np.minimum(np.floor(sc).astype(int), 28), sc - np.minimum(np.floor(sc), 28)
    for i, lab in enumerate(d.label):
        x = np.load(f"{root}/wifi_csi/amp/{lab}.npy").astype(np.float32)        # (T, 3, 3, 30)
        t = np.linspace(0, len(x) - 1, 3000)
        a, f = np.minimum(np.floor(t).astype(int), len(x) - 2), (t - np.minimum(np.floor(t), len(x) - 2))
        x = x[a] * (1 - f)[:, None, None, None] + x[a + 1] * f[:, None, None, None]
        x = x.reshape(300, 10, 3, 3, 30).mean(1)                                   # 100 Hz
        x = x[..., lo] * (1 - w) + x[..., lo + 1] * w                                # 52 subcarriers
        X[i] = x.transpose(1, 2, 3, 0)
        if i % 500 == 0:
            print(f"  loaded {i}/{len(d)}", flush=True)
    np.savez(cache, X=X, act=acts.to_numpy(), env=d.environment.to_numpy())
    return X, acts.to_numpy(), d.environment.to_numpy()


def part_a(X, act, env):
    classes = ["empty", "sit_down", "walk", "wave"]
    keep = np.isin(act, classes)
    y_all = np.array([classes.index(a) if a in classes else -1 for a in act])
    inputs = {"9 pairs": X.reshape(len(X), 9, 52, 300), "2 pairs (cut)": np.stack([X[:, t, r] for t, r in CUT], 1)}
    for name, Xi in inputs.items():
        P = prepare(Xi[keep])
        y_np, e = y_all[keep], env[keep]
        y = torch.from_numpy(y_np.astype(np.int64))
        src = np.where(e == "classroom")[0]
        print(f"\n[{name}] classroom {len(src)} clips, classes {classes}, per class {np.bincount(y_np[src]).tolist()}")
        t0 = time.time()
        within = []
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed)
            fold = np.zeros(len(src), int)
            for c in range(4):
                idx = rng.permutation(np.where(y_np[src] == c)[0])
                fold[idx] = np.arange(len(idx)) % 5
            hits = 0
            for f in range(5):
                tr, te = src[fold != f], src[fold == f]
                m, s = fit(P[tr], y[tr], 4, seed)
                hits += (predict(m, s, P[te]) == y_np[te]).sum()
            within.append(hits / len(src))
        print(f"  within classroom (5-fold): mean {np.mean(within):.1%}, per seed "
              + " ".join(f"{v:.1%}" for v in within) + f"  ({time.time() - t0:.0f} s)")
        for room in ("meeting_room", "empty_room"):
            tgt = np.where(e == room)[0]
            acc = {"source only": [], "AdaBN": [], "SHOT-IM": []}
            for seed in range(SEEDS):
                m, s = fit(P[src], y[src], 4, seed)
                models = {"source only": m, "AdaBN": adabn(m, s, P[tgt]), "SHOT-IM": shot_im(m, s, P[tgt], seed)}
                for k_, mm in models.items():
                    acc[k_].append((predict_proba(mm, s, P[tgt]).argmax(1).numpy() == y_np[tgt]).mean())
            print(f"  classroom -> {room:12} " + "   ".join(
                f"{k_} {np.mean(v):.1%} ({min(v):.1%}-{max(v):.1%})" for k_, v in acc.items()))


def pretrain(Xp, yp, seed):
    torch.manual_seed(seed)
    scale = Xp.std().item()
    m = SmallCNN(2, 10).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=PRE_LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    Xg, yg = Xp.to(DEV), yp.to(DEV)
    for ep in range(PRE_EPOCHS):
        m.train()
        order = torch.randperm(len(yg), device=DEV)
        for i in range(0, len(yg), PRE_BATCH):
            b = order[i:i + PRE_BATCH]
            opt.zero_grad()
            loss_fn(m(Xg[b] / scale), yg[b]).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        acc = np.mean([(m(Xg[i:i + 512] / scale).argmax(1) == yg[i:i + 512]).float().mean().item()
                       for i in range(0, len(yg), 512)])
    return m.cpu(), acc


def transfer(body_model, X, y, k, seed, mode):
    """New final layer on a pretrained body. mode: frozen | frozen_bn | finetune."""
    torch.manual_seed(seed)
    scale = X.std().item()
    m = copy.deepcopy(body_model)
    m.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(48, k))
    if mode == "frozen_bn":
        m = adabn(m, scale, X)              # our training clips only; no labels
    if mode == "finetune":
        opt = torch.optim.Adam([{"params": m.body.parameters(), "lr": 1e-4},
                                {"params": m.head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
    else:
        for p in m.body.parameters():
            p.requires_grad = False
        opt = torch.optim.Adam(m.head.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        if mode == "finetune":
            m.train()
        else:
            m.eval(); m.head.train()       # body keeps its batch-norm averages
        order = torch.randperm(len(y))
        for i in range(0, len(y), BATCH):
            b = order[i:i + BATCH]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss_fn(m(X[b] / scale), y[b]).backward()
            opt.step()
    return m, scale


def part_b(X, act, env, ours):
    classes = ["empty"] + ACTS
    Xp = np.concatenate([np.stack([X[:, t, 0], X[:, t, 2]], 1) for t in range(3)])
    yp = torch.from_numpy(np.tile(np.array([classes.index(a) for a in act]), 3).astype(np.int64))
    Xp = prepare(Xp)
    print(f"\npretraining set: {len(yp)} samples (cut input, 3 transmitters), 10 classes, {PRE_EPOCHS} epochs on {DEV}")
    t0 = time.time()
    bodies = []
    for seed in range(SEEDS):
        m, acc = pretrain(Xp, yp, seed)
        bodies.append(m)
        print(f"  seed {seed}: WiMANS training accuracy {acc:.1%}  ({time.time() - t0:.0f} s)", flush=True)

    d = np.load(ours)
    Xo, yo_np, session = prepare(d["X"]), d["y"], d["session"]
    yo = torch.from_numpy(yo_np.astype(np.int64))
    labels = [str(l) for l in d["labels"]]
    k = len(labels)
    S = sorted(set(session))
    splits = {"leave one session out": [(np.where(session != s)[0], np.where(session == s)[0]) for s in S],
              "sessions 1-3 -> 4": [(np.where(np.isin(session, S[:3]))[0], np.where(session == S[3])[0])]}
    for split, folds in splits.items():
        print(f"\nour data, {split}")
        for mode in ("scratch", "frozen", "frozen_bn", "finetune"):
            accs, right = [], np.zeros(k, int)
            for seed in range(SEEDS):
                hits = n = 0
                for tr, te in folds:
                    if mode == "scratch":
                        m, s = fit(Xo[tr], yo[tr], k, seed)
                    else:
                        m, s = transfer(bodies[seed], Xo[tr], yo[tr], k, seed, mode)
                    p = predict(m, s, Xo[te])
                    hits += (p == yo_np[te]).sum(); n += len(te)
                    right += [(p[yo_np[te] == c] == c).sum() for c in range(k)]
                accs.append(hits / n)
            tot = sum(np.bincount(yo_np[te], minlength=k) for _, te in folds) * SEEDS
            print(f"  {mode:10} mean {np.mean(accs):.1%}, std {np.std(accs):.1%}, per seed "
                  + " ".join(f"{v:.1%}" for v in accs) + "   " + "  ".join(
                      f"{l} {r}/{c}" for l, r, c in zip(labels, right, tot)), flush=True)


def main():
    root, ours = sys.argv[1], sys.argv[2]
    torch.set_num_threads(16)
    t0 = time.time()
    X, act, env = load_wimans(root)
    print(f"WiMANS 2.4 GHz, 0-1 person: {len(X)} clips, rooms {sorted(set(env))}, "
          f"non-finite {int((~np.isfinite(X)).sum())}, zeros {(X == 0).mean():.3%}  ({time.time() - t0:.0f} s)")
    print("\n=== Part A: hardware or physics? learn in the classroom, test in other rooms ===")
    part_a(X, act, env)
    print("\n=== Part B: WiMANS-pretrained backbone on our data ===")
    part_b(X, act, env, ours)
    print(f"\ntotal {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
