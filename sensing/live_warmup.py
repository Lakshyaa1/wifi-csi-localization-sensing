"""
live_warmup.py - streaming AdaBN on the live session, with a start-up warm-up.

    python live_warmup.py <live test session folder> <dataset.npz>

Same windows and models as live_extras.py part B (CNN learned on all lower-sra-room clips, 5 seeds; batch-norm
averages refreshed every 5 s from the windows seen so far, no labels). A live system would first watch the
room for a while before showing answers, so scoring starts after a warm-up of W seconds of the session
(W = 0, 60, 120). Every method is scored on the same windows (those after the warm-up) for each W.
Labels are read only for scoring.
"""
import csv, sys
import numpy as np
import torch
from csi_common import load_raw_log, process_clip
from build_dataset import DATA_SUBCARRIERS
from train_cnn import fit, prepare
from adapt import adabn, predict_proba

FS, N, SEEDS, REFRESH = 100, 500, 5, 10
WARMUPS = (0, 60, 120)


def main():
    sdir, dataset = sys.argv[1], sys.argv[2]
    d = np.load(dataset)
    labels = [str(l) for l in d["labels"]]
    k = len(labels)
    boards = [load_raw_log(f"{sdir}/raw_board{i}.log") for i in (0, 1)]
    rows = list(csv.DictReader(open(f"{sdir}/index.csv", newline="")))
    t_start = float(rows[0]["t_start"])
    W, yl, wt = [], [], []
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
                W.append(np.stack(arr)); yl.append(labels.index(lab) if lab in labels else -1); wt.append(t1 - t_start)
    Xl, yl, wt = prepare(np.stack(W).astype(np.float32)), np.array(yl), np.array(wt)
    print(f"{len(Xl)} windows over {wt[-1]:.0f} s; blocks: " + ", ".join(r["label"].strip() for r in rows))

    X, y = prepare(d["X"]), torch.from_numpy(d["y"].astype(np.int64))
    res = {w: {m: [] for m in ("source", "streaming AdaBN", "full AdaBN")} for w in WARMUPS}
    per = {w: {m: np.zeros((k, 2)) for m in res[w]} for w in WARMUPS}
    for seed in range(SEEDS):
        src, scale = fit(X, y, k, seed)
        src_p = predict_proba(src, scale, Xl).argmax(1).numpy()
        full_p = predict_proba(adabn(src, scale, Xl), scale, Xl).argmax(1).numpy()
        stream = np.zeros(len(Xl), int)
        for s in range(0, len(Xl), REFRESH):
            m = adabn(src, scale, Xl[:max(s, REFRESH)])
            stream[s:s + REFRESH] = predict_proba(m, scale, Xl[s:s + REFRESH]).argmax(1).numpy()
        for w in WARMUPS:
            sel = (yl >= 0) & (wt >= w)
            for name, p in (("source", src_p), ("streaming AdaBN", stream), ("full AdaBN", full_p)):
                res[w][name].append((p[sel] == yl[sel]).mean())
                for c in range(k):
                    per[w][name][c] += [(p[sel][yl[sel] == c] == c).sum(), (yl[sel] == c).sum()]
    for w in WARMUPS:
        sel = (yl >= 0) & (wt >= w)
        print(f"\nwarm-up {w} s: {sel.sum()} scored windows")
        for name, v in res[w].items():
            print(f"  {name:16} mean {np.mean(v):.1%}, per seed " + " ".join(f"{a:.1%}" for a in v) + "   "
                  + "  ".join(f"{l} {per[w][name][c, 0] / max(per[w][name][c, 1], 1):.0%}" for c, l in enumerate(labels)))


if __name__ == "__main__":
    main()
