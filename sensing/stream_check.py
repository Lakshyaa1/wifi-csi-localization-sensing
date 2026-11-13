"""
stream_check.py - do online adaptation methods survive realistic streams on our ESP32 data?

    python stream_check.py <live test session folder> <dataset.npz>

Part 1 - streams (CNN learned on all lower-sra-room clips, 5 seeds; live session replayed in time order,
5 s windows; labels read only for scoring). Three streams built from the same recorded windows:
  blocks:        every window (0.5 s apart), 30 s blocks of one activity, as recorded
  mostly empty:  every empty window, but only every 5th window of everything else
  no walk:       as recorded, with the walk blocks removed (one activity never happens)
Methods, all label-free, all processing the stream 10 windows at a time, in order:
  source          no adaptation
  AdaBN all       batch-norm averages from every window seen so far (ours so far)
  AdaBN recent    batch-norm averages from the previous 20 windows only
  AdaBN running   running average updated one window at a time (momentum 0.05)
  AdaBN blend     half source averages, half "AdaBN all" averages
  TENT            batch-norm averages from the current 10 windows, and batch-norm scale/shift trained to
                  lower prediction entropy (Adam, lr 1e-3, one step per 10 windows)
  T3A             network untouched; each class's final-layer weight is replaced by the average feature of
                  the windows assigned to it so far (20 most confident per class kept)
Reported: share of windows right, and the average of the per-activity shares (fair when one activity dominates).

Part 2 - offline adaptation on unbalanced data (CNN learned on sessions 1-3, session 4 clips):
  adapt AdaBN / SHOT-IM on: all 39 clips; empty-heavy (10 empty + 3 of each other); no walk clips.
  Scored on all 39 clips (the adaptation clips are among them).
Settings fixed before running.
"""
import copy, csv, sys
import numpy as np
import torch
from torch import nn
from csi_common import load_raw_log, process_clip
from build_dataset import DATA_SUBCARRIERS
from train_cnn import fit, prepare
from adapt import adabn, shot_im, predict_proba

FS, N, SEEDS, STEP = 100, 500, 5, 10


def bn_layers(m):
    return [l for l in m.modules() if isinstance(l, nn.BatchNorm2d)]


@torch.no_grad()
def feats(m, scale, X):
    m.eval()
    return m.body(X / scale).mean(dim=2).mean(dim=-1)


def run_stream(src, scale, X, method, k, seed):
    """Predictions for stream X (in order), STEP windows at a time."""
    torch.manual_seed(seed)
    out = np.zeros(len(X), int)
    if method == "AdaBN running":
        m = copy.deepcopy(src)
        for l in bn_layers(m):
            l.momentum = 0.05
    if method == "TENT":
        m = copy.deepcopy(src)
        params = [p for l in bn_layers(m) for p in (l.weight, l.bias)]
        for p in m.parameters():
            p.requires_grad = False
        for p in params:
            p.requires_grad = True
        opt = torch.optim.Adam(params, lr=1e-3)
    if method == "T3A":
        W = src.head[1].weight.detach()
        sup = W / W.norm(dim=1, keepdim=True)
        sup_y = torch.eye(k)
        sup_ent = torch.zeros(k)
    for s in range(0, len(X), STEP):
        xb = X[s:s + STEP]
        if method == "source":
            p = predict_proba(src, scale, xb)
        elif method == "AdaBN all":
            p = predict_proba(adabn(src, scale, X[:max(s, STEP)]), scale, xb)
        elif method == "AdaBN recent":
            p = predict_proba(adabn(src, scale, X[max(0, s - 2 * STEP):s] if s else xb), scale, xb)
        elif method == "AdaBN blend":
            a = adabn(src, scale, X[:max(s, STEP)])
            for la, ls in zip(bn_layers(a), bn_layers(src)):
                la.running_mean.copy_(0.5 * la.running_mean + 0.5 * ls.running_mean)
                la.running_var.copy_(0.5 * la.running_var + 0.5 * ls.running_var)
            p = predict_proba(a, scale, xb)
        elif method == "AdaBN running":
            ps = []
            for x in xb:
                with torch.no_grad():
                    m.body.train(); m.body(x[None] / scale)
                ps.append(predict_proba(m, scale, x[None]))
            p = torch.cat(ps)
        elif method == "TENT":
            if len(xb) < 2:
                p = predict_proba(m, scale, xb)
            else:
                m.body.train(); m.head.eval()
                p = torch.softmax(m(xb / scale), 1)
                loss = -(p * torch.log(p + 1e-6)).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                p = p.detach()
        elif method == "T3A":
            z = feats(src, scale, xb)
            p0 = predict_proba(src, scale, xb)
            ps = []
            for zi, pi in zip(z, p0):
                sup = torch.cat([sup, (zi / zi.norm())[None]])
                sup_y = torch.cat([sup_y, torch.eye(k)[pi.argmax()][None]])
                sup_ent = torch.cat([sup_ent, -(pi * torch.log(pi + 1e-6)).sum()[None]])
                keep = []
                for c in range(k):
                    idx = torch.where(sup_y[:, c] == 1)[0]
                    keep.append(idx[sup_ent[idx].argsort()[:20]])
                keep = torch.cat(keep)
                sup, sup_y, sup_ent = sup[keep], sup_y[keep], sup_ent[keep]
                cent = sup_y.T @ sup
                cent = cent / cent.norm(dim=1, keepdim=True)
                ps.append((zi @ cent.T)[None])
            p = torch.cat(ps)
        out[s:s + STEP] = p.argmax(1).numpy()
    return out


def main():
    sdir, dataset = sys.argv[1], sys.argv[2]
    d = np.load(dataset)
    labels = [str(l) for l in d["labels"]]
    k = len(labels)
    boards = [load_raw_log(f"{sdir}/raw_board{i}.log") for i in (0, 1)]
    rows = list(csv.DictReader(open(f"{sdir}/index.csv", newline="")))
    W, yl = [], []
    for r in rows:
        a, b, lab = float(r["t_start"]), float(r["t_end"]), r["label"].strip()
        for t1 in np.arange(a + 5, b + 1e-6, 0.5):
            arr = []
            for bd in boards:
                A, info = process_clip(bd, t1 - 5, t1, DATA_SUBCARRIERS, fs=FS, n=N)
                if A is None or not info["ok"]:
                    break
                arr.append(A.T)
            if len(arr) == 2:
                W.append(np.stack(arr)); yl.append(labels.index(lab) if lab in labels else -1)
    Xl, yl = prepare(np.stack(W).astype(np.float32)), np.array(yl)
    e, wk = labels.index("empty"), labels.index("walk")
    idx = np.arange(len(yl))
    other_rank = np.cumsum(yl != e) - 1
    streams = {"blocks": idx,
               "mostly empty": idx[(yl == e) | (other_rank % 5 == 0)],
               "no walk": idx[yl != wk]}
    X, y = prepare(d["X"]), torch.from_numpy(d["y"].astype(np.int64))
    methods = ["source", "AdaBN all", "AdaBN recent", "AdaBN running", "AdaBN blend", "TENT", "T3A"]

    print("Part 1 - streams from the live session, CNN learned on all 152 clips, 5 seeds")
    sources = [fit(X, y, k, seed) for seed in range(SEEDS)]
    for sname, sel in streams.items():
        ys = yl[sel]
        known = ys >= 0
        mix = "  ".join(f"{l} {np.mean(ys[known] == c):.0%}" for c, l in enumerate(labels))
        print(f"\n[{sname}] {len(sel)} windows, {known.sum()} scored; mix: {mix}")
        print(f"  {'method':14} {'right':>6} {'per-activity avg':>17}   per activity")
        for meth in methods:
            acc, bal, per = [], [], np.zeros((k, 2))
            for seed, (src, scale) in enumerate(sources):
                p = run_stream(src, scale, Xl[sel], meth, k, seed)
                acc.append((p[known] == ys[known]).mean())
                pc = [(p[ys == c] == c).mean() for c in range(k) if (ys == c).any()]
                bal.append(np.mean(pc))
                for c in range(k):
                    per[c] += [(p[ys == c] == c).sum(), (ys == c).sum()]
            print(f"  {meth:14} {np.mean(acc):6.1%} {np.mean(bal):17.1%}   "
                  + "  ".join(f"{l} {per[c, 0] / per[c, 1]:.0%}" for c, l in enumerate(labels) if per[c, 1]), flush=True)

    print("\nPart 2 - offline adaptation on unbalanced clips: learn on sessions 1-3, session 4 (39 clips), 5 seeds")
    S = sorted(set(d["session"]))
    tr, te = np.where(np.isin(d["session"], S[:3]))[0], np.where(d["session"] == S[3])[0]
    yt = d["y"][te]
    rng = np.random.default_rng(0)
    heavy = np.concatenate([np.where(yt == e)[0]] + [rng.permutation(np.where(yt == c)[0])[:3] for c in range(k) if c != e])
    subsets = {"all 39 clips (balanced)": np.arange(len(te)), "empty-heavy (10 empty + 3 each)": heavy,
               "no walk clips": np.where(yt != wk)[0]}
    res = {(s, m): [] for s in subsets for m in ("source", "AdaBN", "SHOT-IM")}
    per = {key: np.zeros((k, 2)) for key in res}
    for seed in range(SEEDS):
        src, scale = fit(X[tr], y[tr], k, seed)
        for sname, sub in subsets.items():
            Xa = X[te][sub]
            models = {"source": src, "AdaBN": adabn(src, scale, Xa), "SHOT-IM": shot_im(src, scale, Xa, seed)}
            for m, mod in models.items():
                p = predict_proba(mod, scale, X[te]).argmax(1).numpy()
                res[(sname, m)].append((p == yt).mean())
                for c in range(k):
                    per[(sname, m)][c] += [(p[yt == c] == c).sum(), (yt == c).sum()]
    for sname in subsets:
        print(f"  adapted on {sname}")
        for m in ("source", "AdaBN", "SHOT-IM"):
            v, pc = res[(sname, m)], per[(sname, m)]
            print(f"    {m:8} {np.mean(v):6.1%} ({min(v):.1%}-{max(v):.1%})   "
                  + "  ".join(f"{l} {pc[c, 0] / pc[c, 1]:.0%}" for c, l in enumerate(labels)))


if __name__ == "__main__":
    main()
