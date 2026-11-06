"""
calibration_check.py - does a short labelled calibration from the same session fix the live classifier?

    python calibration_check.py <live test session folder> <dataset.npz>

The live test has two blocks of each action. Calibrate on one block of each action, score every 5 s window
(every 0.5 s) of the other blocks, then swap the roles and average. Calibration clips are 5 s windows every
1 s inside the calibration blocks, with their labels (this uses new-session labels, so it is a demo
setting, not source-free adaptation). Compared:
  no calibration:   CNN learned on sessions 1-3, as is / with AdaBN on the calibration blocks (no labels)
  calibrated LR:    logistic regression learned on the calibration clips only
  LR + history:     logistic regression learned on sessions 1-4 plus the calibration clips
  calibrated CNN:   CNN learned on sessions 1-3, then fine-tuned on the calibration clips
Settings fixed beforehand: fine-tuning with Adam, lr 1e-4, weight decay 1e-4, 50 epochs, batch 16.
"""
import copy, csv, sys
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from csi_common import load_raw_log
from train_baseline import features
from train_cnn import fit, prepare
from adapt import adabn, predict_proba
from adapt_live_test import cut


def windows(boards, a, b, step):
    out = []
    for t1 in np.arange(a + 5, b + 1e-6, step):
        c = cut(boards, t1 - 5, t1)
        if c is not None:
            out.append(c)
    return out


def finetune(model, scale, X, y, seed):
    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(50):
        m.train()
        order = torch.randperm(len(y))
        for i in range(0, len(y), 16):
            b = order[i:i + 16]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss_fn(m(X[b] / scale), y[b]).backward()
            opt.step()
    return m


def lr_fit(F, y):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(F, y)


def main():
    sdir, dataset = sys.argv[1], sys.argv[2]
    d = np.load(dataset)
    labels = [str(l) for l in d["labels"]]
    k = len(labels)
    S = sorted(set(d["session"]))
    tr3 = np.where(np.isin(d["session"], S[:3]))[0]
    Xd, yd = prepare(d["X"]), torch.from_numpy(d["y"].astype(np.int64))
    Fd = np.stack([features(c, 100) for c in d["X"]])

    boards = [load_raw_log(f"{sdir}/raw_board{i}.log") for i in (0, 1)]
    rows = [r for r in csv.DictReader(open(f"{sdir}/index.csv", newline="")) if r["label"].strip() in labels]
    blocks = {l: [(float(r["t_start"]), float(r["t_end"])) for r in rows if r["label"].strip() == l] for l in labels}
    assert all(len(v) == 2 for v in blocks.values()), blocks

    sources = [fit(Xd[tr3], yd[tr3], k, seed) for seed in range(5)]
    results = {}
    per_class = {}
    for cal in (0, 1):
        test = 1 - cal
        cw, cy, tw, ty = [], [], [], []
        for i, l in enumerate(labels):
            w = windows(boards, *blocks[l][cal], step=1.0); cw += w; cy += [i] * len(w)
            w = windows(boards, *blocks[l][test], step=0.5); tw += w; ty += [i] * len(w)
        cy, ty = np.array(cy), np.array(ty)
        Xc, Xt = prepare(np.stack(cw)), prepare(np.stack(tw))
        Fc, Ft = np.stack([features(c, 100) for c in cw]), np.stack([features(c, 100) for c in tw])
        print(f"calibrate on block {cal + 1} of each action ({len(cy)} clips), score block {test + 1} ({len(ty)} windows)")

        preds = {"LR, calibration only": [lr_fit(Fc, cy).predict(Ft)],
                 "LR, sessions 1-4 + calibration": [lr_fit(np.vstack([Fd, Fc]), np.r_[d["y"], cy]).predict(Ft)]}
        for name in ("CNN, no calibration", "CNN + AdaBN (no labels)", "CNN, fine-tuned on calibration"):
            preds[name] = []
        for seed, (src, scale) in enumerate(sources):
            preds["CNN, no calibration"].append(predict_proba(src, scale, Xt).argmax(1).numpy())
            preds["CNN + AdaBN (no labels)"].append(predict_proba(adabn(src, scale, Xc), scale, Xt).argmax(1).numpy())
            ft = finetune(src, scale, Xc, torch.from_numpy(cy.astype(np.int64)), seed)
            preds["CNN, fine-tuned on calibration"].append(predict_proba(ft, scale, Xt).argmax(1).numpy())
        for name, ps in preds.items():
            results.setdefault(name, []).extend((p == ty).mean() for p in ps)
            pc = per_class.setdefault(name, np.zeros((k, 2)))
            for p in ps:
                for c in range(k):
                    pc[c] += [(p[ty == c] == c).sum(), (ty == c).sum()]

    print(f"\n{'method':34} {'mean':>6} {'range':>14}   right per action (both directions, all seeds)")
    for name, v in results.items():
        pc = per_class[name]
        print(f"{name:34} {np.mean(v):6.1%} {min(v):6.1%}-{max(v):5.1%}   "
              + "  ".join(f"{l} {pc[c, 0] / pc[c, 1]:.0%}" for c, l in enumerate(labels)))


if __name__ == "__main__":
    main()
