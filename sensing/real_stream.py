"""
real_stream.py - online adaptation on a stream that was really recorded mostly empty.

    python real_stream.py <realistic session folder> <dataset.npz>

Same models and methods as stream_check.py part 1 (CNN learned on all lower-sra-room clips, 5 seeds; all
methods label-free, 10 windows at a time, in order). The stream is the session exactly as recorded
(record.py --script: long empty blocks with short activities between), 5 s windows every 0.5 s inside each
block; windows with a stall on either board are left out. Labels are read only for scoring.
"""
import csv, sys
import numpy as np
import torch
from csi_common import load_raw_log, process_clip
from build_dataset import DATA_SUBCARRIERS
from train_cnn import fit, prepare
from stream_check import run_stream, FS, N, SEEDS


def main():
    sdir, dataset = sys.argv[1], sys.argv[2]
    d = np.load(dataset)
    labels = [str(l) for l in d["labels"]]
    k = len(labels)
    boards = [load_raw_log(f"{sdir}/raw_board{i}.log") for i in (0, 1)]
    W, yl = [], []
    for r in csv.DictReader(open(f"{sdir}/index.csv", newline="")):
        a, b, lab = float(r["t_start"]), float(r["t_end"]), r["label"].strip()
        for t1 in np.arange(a + 5, b + 1e-6, 0.5):
            arr = []
            for bd in boards:
                A, info = process_clip(bd, t1 - 5, t1, DATA_SUBCARRIERS, fs=FS, n=N)
                if A is None or not info["ok"]:
                    break
                arr.append(A.T)
            if len(arr) == 2:
                W.append(np.stack(arr)); yl.append(labels.index(lab))
    Xl, yl = prepare(np.stack(W).astype(np.float32)), np.array(yl)
    X, y = prepare(d["X"]), torch.from_numpy(d["y"].astype(np.int64))
    print(f"{len(yl)} windows; mix: " + "  ".join(f"{l} {np.mean(yl == c):.0%}" for c, l in enumerate(labels)))
    sources = [fit(X, y, k, seed) for seed in range(SEEDS)]
    print(f"  {'method':14} {'right':>6} {'range over seeds':>17} {'per-activity avg':>17}   per activity   | share of answers")
    for meth in ["source", "AdaBN all", "AdaBN recent", "AdaBN running", "AdaBN blend", "TENT", "T3A"]:
        acc, bal, per, said = [], [], np.zeros((k, 2)), np.zeros(k)
        for seed, (src, scale) in enumerate(sources):
            p = run_stream(src, scale, Xl, meth, k, seed)
            acc.append((p == yl).mean())
            bal.append(np.mean([(p[yl == c] == c).mean() for c in range(k)]))
            for c in range(k):
                per[c] += [(p[yl == c] == c).sum(), (yl == c).sum()]
                said[c] += (p == c).sum()
        print(f"  {meth:14} {np.mean(acc):6.1%} {min(acc):8.1%}-{max(acc):.1%} {np.mean(bal):17.1%}   "
              + "  ".join(f"{l} {per[c, 0] / per[c, 1]:.0%}" for c, l in enumerate(labels))
              + "   | " + "  ".join(f"{l} {said[c] / said.sum():.0%}" for c, l in enumerate(labels)), flush=True)


if __name__ == "__main__":
    main()
