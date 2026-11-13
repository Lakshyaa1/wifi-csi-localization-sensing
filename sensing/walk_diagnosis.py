"""
walk_diagnosis.py - why is walk called wave on later sessions? No model is trained here.

    python walk_diagnosis.py <dataset.npz> <session folder> [<session folder> ...]

For every training clip and every 5 s window of the given cued sessions, per board, on the model's own input
(relative change per subcarrier):
  strength   median over subcarriers of the standard deviation over time
  fast share share of the movement energy above 2 Hz (median over subcarriers)
Printed per source and activity: the median, with the 25th-75th percentile range.
"""
import csv, sys
import numpy as np
from csi_common import load_raw_log, process_clip
from build_dataset import DATA_SUBCARRIERS


def stats(X):   # X: (n, boards, subcarriers, time at 100 Hz)
    R = X / (X.mean(-1, keepdims=True) + 1e-6) - 1
    strength = np.median(R.std(-1), axis=-1)                       # (n, boards)
    P = np.abs(np.fft.rfft(R - R.mean(-1, keepdims=True), axis=-1)) ** 2
    f = np.fft.rfftfreq(R.shape[-1], 0.01)
    fast = np.median(P[..., f > 2].sum(-1) / (P[..., f > 0].sum(-1) + 1e-12), axis=-1)
    return strength, fast


def show(name, lab, s, f):
    q = lambda v: f"{np.median(v):.3f} ({np.percentile(v, 25):.3f}-{np.percentile(v, 75):.3f})"
    print(f"  {name:22} {lab:5} n={len(s):3}  strength b0 {q(s[:, 0])}  b1 {q(s[:, 1])}   fast share b0 {q(f[:, 0])}  b1 {q(f[:, 1])}")


def main():
    d = np.load(sys.argv[1])
    labels = [str(l) for l in d["labels"]]
    s, f = stats(d["X"].astype(np.float64))
    print("training clips, by session")
    for i, ses in enumerate(sorted(set(d["session"]))):
        for c, l in enumerate(labels):
            m = (d["session"] == ses) & (d["y"] == c)
            show(f"S{i + 1} {ses[8:16]}", l, s[m], f[m])
    print("training clips, all sessions")
    for c, l in enumerate(labels):
        show("S1-S4", l, s[d["y"] == c], f[d["y"] == c])
    for sd in sys.argv[2:]:
        boards = [load_raw_log(f"{sd}/raw_board{i}.log") for i in (0, 1)]
        print(sd.rstrip("/").split("session_")[-1] + ", per block")
        allw = {}
        for r in csv.DictReader(open(f"{sd}/index.csv", newline="")):
            a, b, lab = float(r["t_start"]), float(r["t_end"]), r["label"].strip()
            W = []
            for t1 in np.arange(a + 5, b + 1e-6, 0.5):
                arr = []
                for bd in boards:
                    A, info = process_clip(bd, t1 - 5, t1, DATA_SUBCARRIERS, fs=100, n=500)
                    if A is None or not info["ok"]:
                        break
                    arr.append(A.T)
                if len(arr) == 2:
                    W.append(np.stack(arr))
            if W:
                ss, ff = stats(np.stack(W).astype(np.float64))
                if lab != "empty":
                    show(f"block {r['clip_id']}", lab, ss, ff)
                allw.setdefault(lab, []).append((ss, ff))
        for lab, v in allw.items():
            show("whole session", lab, np.concatenate([x[0] for x in v]), np.concatenate([x[1] for x in v]))


if __name__ == "__main__":
    main()
