"""
adapt_live_test.py - adapt the CNN to a scripted live-test session (no labels), then score its windows.

    python adapt_live_test.py <live test session folder> <dataset.npz>

1. Learn the CNN on sessions 1-3 of dataset.npz (5 seeds).
2. Cut every block of the live test into non-overlapping 5 s clips, all blocks including ones with labels the
   model doesn't know (e.g. 'seated'). These clips, without labels, are the adaptation data.
3. Score every 5 s window (every 0.5 s) in the blocks of known actions, for: source only, AdaBN, SHOT-IM.
Labels are read only for scoring. Settings come from adapt.py and were fixed beforehand.
"""
import csv, sys, collections
import numpy as np
import torch
from csi_common import load_raw_log, process_clip
from build_dataset import DATA_SUBCARRIERS
from train_cnn import fit, prepare
from adapt import adabn, shot_im, predict_proba, change


def cut(boards, t0, t1, n=500, fs=100):
    arrs = []
    for b in boards:
        A, info = process_clip(b, t0, t1, DATA_SUBCARRIERS, fs=fs, n=n)
        if A is None or not info["ok"]:
            return None
        arrs.append(A.T)
    return np.stack(arrs)


def main():
    sdir, dataset = sys.argv[1], sys.argv[2]
    d = np.load(dataset)
    labels = [str(l) for l in d["labels"]]
    k = len(labels)
    S = sorted(set(d["session"]))
    tr = np.where(np.isin(d["session"], S[:3]))[0]
    X, y = prepare(d["X"]), torch.from_numpy(d["y"].astype(np.int64))

    boards = [load_raw_log(f"{sdir}/raw_board{i}.log") for i in (0, 1)]
    rows = list(csv.DictReader(open(f"{sdir}/index.csv", newline="")))
    adapt_clips, win, win_label = [], [], []
    for r in rows:
        a, b = float(r["t_start"]), float(r["t_end"])
        for t0 in np.arange(a, b - 5 + 1e-6, 5.0):
            c = cut(boards, t0, t0 + 5)
            if c is not None:
                adapt_clips.append(c)
        label = r["label"].strip()
        if label in labels:
            for t1 in np.arange(a + 5, b + 1e-6, 0.5):
                c = cut(boards, t1 - 5, t1)
                if c is not None:
                    win.append(c); win_label.append(labels.index(label))
    Xa, Xw, yw = prepare(np.stack(adapt_clips)), prepare(np.stack(win)), np.array(win_label)
    print(f"adaptation clips (unlabelled, all blocks): {len(Xa)};  scored windows (known actions): {len(Xw)}")

    methods = ["source only", "AdaBN", "SHOT-IM"]
    acc = {m: [] for m in methods}
    conf = {m: np.zeros((k, k), int) for m in methods}
    for seed in range(5):
        src, scale = fit(X[tr], y[tr], k, seed)
        models = {"source only": src, "AdaBN": adabn(src, scale, Xa), "SHOT-IM": shot_im(src, scale, Xa, seed)}
        print(f"  seed {seed}: SHOT-IM changed feature layers by {change(models['SHOT-IM'].body, src.body):.1%}, "
              f"final layer by {change(models['SHOT-IM'].head, src.head):.1%}")
        for m, mod in models.items():
            p = predict_proba(mod, scale, Xw).argmax(1).numpy()
            acc[m].append((p == yw).mean())
            for t, g in zip(yw, p):
                conf[m][t, g] += 1
    print(f"\n{'method':12} {'mean':>6}  per seed")
    for m in methods:
        print(f"{m:12} {np.mean(acc[m]):6.1%}  " + " ".join(f"{v:.1%}" for v in acc[m]))
    for m in methods:
        print(f"\n{m}: rows true, share of windows given each label {labels}")
        for i, l in enumerate(labels):
            row = conf[m][i] / conf[m][i].sum()
            print(f"  {l:6} " + "  ".join(f"{labels[j]} {row[j]:.0%}" for j in range(k)))


if __name__ == "__main__":
    main()
