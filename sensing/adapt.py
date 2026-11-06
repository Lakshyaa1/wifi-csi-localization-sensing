"""
adapt.py - does label-free adaptation recover the drop on a new session or room?

    python adapt.py dataset.npz --train 1 2 3 --test 4

For each seed: learn the CNN (train_cnn.py settings) on the --train sessions, then on the --test clips:
  1. source only: no change
  2. AdaBN:      re-measure every batch-norm layer's average and spread on the test clips
  3. SHOT-IM:    keep the final layer fixed, retrain the earlier layers to make each prediction sure
                 (low entropy) while keeping the average prediction spread over all classes
Adaptation sees only the test clips' CSI. Their labels are used once, at the end, for scoring.
Settings are fixed in advance (Adam, lr 1e-4, weight decay 1e-4, 50 epochs, as in the MU-SHOT-Fi
code's target stage) and the last epoch is scored.
"""
import argparse, copy
import numpy as np
import torch
from torch import nn
from train_cnn import fit, prepare

ADAPT_LR, ADAPT_WD, ADAPT_EPOCHS, ADAPT_BATCH = 1e-4, 1e-4, 50, 16


@torch.no_grad()
def predict_proba(model, scale, X):
    model.eval()
    return torch.softmax(model(X / scale), 1)


@torch.no_grad()
def adabn(model, scale, X):
    m = copy.deepcopy(model)
    for layer in m.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.reset_running_stats()
            layer.momentum = None          # plain average over everything seen
    m.train()
    m(X / scale)                           # one pass over all test clips; no weights change
    return m


def shot_im(model, scale, X, seed):
    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    for p in m.head.parameters():
        p.requires_grad = False            # the classifier keeps room A's meaning of each class
    opt = torch.optim.Adam(m.body.parameters(), lr=ADAPT_LR, weight_decay=ADAPT_WD)
    for _ in range(ADAPT_EPOCHS):
        m.body.train()
        m.head.eval()
        order = torch.randperm(len(X))
        for i in range(0, len(X), ADAPT_BATCH):
            b = order[i:i + ADAPT_BATCH]
            if len(b) < 2:
                continue
            p = torch.softmax(m(X[b] / scale), 1)
            sure = -(p * torch.log(p + 1e-6)).sum(1).mean()          # average entropy per clip
            pbar = p.mean(0)
            varied = -(pbar * torch.log(pbar + 1e-6)).sum()          # entropy of the average
            opt.zero_grad()
            (sure - varied).backward()
            opt.step()
    return m


def change(a, b):
    """Relative change of all weights in module a vs b."""
    num = sum(((p - q) ** 2).sum() for p, q in zip(a.parameters(), b.parameters()))
    den = sum((q ** 2).sum() for q in b.parameters())
    return float((num / den).sqrt())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--train", type=int, nargs="+", required=True)
    ap.add_argument("--test", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--holdout", action="store_true",
                    help="adapt on half of the test clips and score the other half, then swap the halves")
    a = ap.parse_args()
    d = np.load(a.dataset)
    X, y_np, session = prepare(d["X"]), d["y"], d["session"]
    labels = [str(l) for l in d["labels"]]
    k = len(labels)
    S = sorted(set(session))
    pick = lambda nums: np.where(np.isin(session, [S[i - 1] for i in nums]))[0]
    tr, te = pick(a.train), pick(a.test)
    y = torch.from_numpy(y_np.astype(np.int64))
    Xt = X[te]                                   # test CSI only; labels stay in y_test below
    y_test = y_np[te]
    print(f"learn on sessions {a.train} ({len(tr)} clips), adapt to and test on {a.test} ({len(te)} clips)")
    print(f"adaptation: Adam lr {ADAPT_LR}, weight decay {ADAPT_WD}, {ADAPT_EPOCHS} epochs, batch {ADAPT_BATCH}")

    methods = ["source only", "AdaBN", "SHOT-IM"]
    acc = {m: [] for m in methods}
    right = {m: np.zeros(k, int) for m in methods}
    share = {m: np.zeros(k) for m in methods}
    for seed in range(a.seeds):
        src, scale = fit(X[tr], y[tr], k, seed)
        if a.holdout:
            # random halves, balanced by class; the split uses labels only to pick clips, never to adapt
            rng = np.random.default_rng(100 + seed)
            half = np.zeros(len(te), bool)
            for c in range(k):
                idx = rng.permutation(np.where(y_test == c)[0])
                half[idx[:len(idx) // 2 + rng.integers(2) * (len(idx) % 2)]] = True
            folds = [(half, ~half), (~half, half)]
        else:
            folds = [(np.ones(len(te), bool), np.ones(len(te), bool))]
        preds = {m: np.zeros(len(te), int) for m in methods}
        for adapt_on, score_on in folds:
            models = {"source only": src, "AdaBN": adabn(src, scale, Xt[adapt_on]),
                      "SHOT-IM": shot_im(src, scale, Xt[adapt_on], seed)}
            sm = models["SHOT-IM"]
            print(f"  seed {seed}: adapted on {adapt_on.sum()} clips, scored on {score_on.sum()}; SHOT-IM changed "
                  f"feature layers by {change(sm.body, src.body):.1%}, final layer by {change(sm.head, src.head):.1%}")
            for name, m in models.items():
                preds[name][score_on] = predict_proba(m, scale, Xt[score_on]).argmax(1).numpy()
        for name in methods:
            pred = preds[name]
            share[name] += np.bincount(pred, minlength=k) / len(pred) / a.seeds
            acc[name].append((pred == y_test).mean())                     # labels used only here
            right[name] += np.array([(pred[y_test == c] == c).sum() for c in range(k)])

    n = np.bincount(y_test, minlength=k) * a.seeds
    print(f"\n{'method':12} {'mean':>6} {'std':>5}  per seed{'':16}  right per class (summed over seeds)")
    for m in methods:
        print(f"{m:12} {np.mean(acc[m]):6.1%} {np.std(acc[m]):5.1%}  " + " ".join(f"{v:.1%}" for v in acc[m])
              + "   " + "  ".join(f"{l} {r}/{c}" for l, r, c in zip(labels, right[m], n)))
    print("\nshare of test clips given each label (no labels needed for this)")
    for m in methods:
        print(f"  {m:12} " + "  ".join(f"{l} {s:.0%}" for l, s in zip(labels, share[m])))


if __name__ == "__main__":
    main()
