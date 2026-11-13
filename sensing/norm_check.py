"""
norm_check.py - does dividing each board's signal by that board's own typical strength bring walk back?

    python norm_check.py <dataset.npz> <cued session folder> [...]

Strength of a 5 s clip on one board = median over subcarriers of the standard deviation over time of the
model input (relative change). A board's reference level in one session is a percentile of its clips' strengths:
  none    no normalisation (as before)
  floor   10th percentile  (the quiet level)
  median  50th percentile
  top     95th percentile  (the strongest movement seen)
Training sessions: the percentile over the whole session (no labels used). Replayed sessions: causal - over
the windows seen so far, refreshed every 10 windows (the first 10 use themselves). Each board's input is
divided by its reference before the CNN.
Part 1: CNN learned on all training clips (5 seeds), each session replayed in order; no adaptation and
        AdaBN blend (stream_check.py). Part 2: leave one training session out, no adaptation.
Settings fixed before running; all variants are reported.
"""
import csv, sys, time
import numpy as np
import torch
from csi_common import load_raw_log, process_clip
from build_dataset import DATA_SUBCARRIERS
from train_cnn import fit, prepare
from adapt import predict_proba
from stream_check import run_stream, SEEDS

PCT = {"none": None, "floor": 10, "median": 50, "top": 95}


def strength(P):   # P: prepared tensor (n, boards, subcarriers, time) -> (n, boards)
    return P.std(dim=-1).median(dim=-1).values.numpy()


def norm_session(P, pct):
    if pct is None:
        return P
    ref = np.percentile(strength(P), pct, axis=0)
    return P / torch.tensor(ref, dtype=P.dtype)[None, :, None, None]


def norm_stream(P, pct):
    if pct is None:
        return P
    s, out = strength(P), P.clone()
    for i in range(0, len(P), 10):
        ref = np.percentile(s[:max(i, 10)], pct, axis=0)
        out[i:i + 10] = P[i:i + 10] / torch.tensor(ref, dtype=P.dtype)[None, :, None, None]
    return out


def windows(sd, labels):
    boards = [load_raw_log(f"{sd}/raw_board{i}.log") for i in (0, 1)]
    W, yl = [], []
    for r in csv.DictReader(open(f"{sd}/index.csv", newline="")):
        a, b, lab = float(r["t_start"]), float(r["t_end"]), r["label"].strip()
        if r["flag"].strip().startswith(("interrupted", "wrong_action")):   # marked by hand: not what was cued
            continue
        for t1 in np.arange(a + 5, b + 1e-6, 0.5):
            arr = []
            for bd in boards:
                A, info = process_clip(bd, t1 - 5, t1, DATA_SUBCARRIERS, fs=100, n=500)
                if A is None or not info["ok"]:
                    break
                arr.append(A.T)
            if len(arr) == 2:
                W.append(np.stack(arr)); yl.append(labels.index(lab) if lab in labels else -1)
    return prepare(np.stack(W).astype(np.float32)), np.array(yl)


def main():
    d = np.load(sys.argv[1])
    labels = [str(l) for l in d["labels"]]
    k = len(labels)
    P, y_np, ses = prepare(d["X"]), d["y"], d["session"]
    y = torch.from_numpy(y_np.astype(np.int64))
    S = sorted(set(ses))
    streams = {sd.rstrip("/").split("session_")[-1]: windows(sd, labels) for sd in sys.argv[2:]}
    train = {}
    for name, pct in PCT.items():
        X = P.clone()
        for s_ in S:
            m = torch.from_numpy(ses == s_)
            X[m] = norm_session(P[m], pct)
        train[name] = X

    print("Part 1 - learn on all training clips, replay each session in order (5 seeds)")
    t0 = time.time()
    models = {name: [fit(train[name], y, k, seed) for seed in range(SEEDS)] for name in PCT}
    print(f"  ({len(PCT) * SEEDS} models in {time.time() - t0:.0f} s)")
    for sname, (Xs, ys) in streams.items():
        known = ys >= 0
        print(f"\n[{sname}] {known.sum()} scored windows; mix: " + "  ".join(f"{l} {np.mean(ys[known] == c):.0%}" for c, l in enumerate(labels)))
        print(f"  {'input':7} {'method':12} {'right':>6} {'per-activity avg':>17}   per activity")
        for name, pct in PCT.items():
            Xn = norm_stream(Xs, pct)
            for meth in ("source", "AdaBN blend"):
                acc, bal, per = [], [], np.zeros((k, 2))
                for seed, (src, scale) in enumerate(models[name]):
                    p = run_stream(src, scale, Xn, meth, k, seed)
                    acc.append((p[known] == ys[known]).mean())
                    bal.append(np.mean([(p[ys == c] == c).mean() for c in range(k) if (ys == c).any()]))
                    for c in range(k):
                        per[c] += [(p[ys == c] == c).sum(), (ys == c).sum()]
                print(f"  {name:7} {meth:12} {np.mean(acc):6.1%} {np.mean(bal):17.1%}   "
                      + "  ".join(f"{l} {per[c, 0] / per[c, 1]:.0%}" for c, l in enumerate(labels) if per[c, 1]), flush=True)

    print("\nPart 2 - leave one training session out, no adaptation (5 seeds); test session normalised over itself")
    for name in PCT:
        accs, per = [], np.zeros((k, 2))
        for seed in range(SEEDS):
            right = 0
            for s_ in S:
                te, tr = ses == s_, ses != s_
                src, scale = fit(train[name][tr], y[tr], k, seed)
                p = predict_proba(src, scale, train[name][te]).argmax(1).numpy()
                right += (p == y_np[te]).sum()
                for c in range(k):
                    per[c] += [(p[y_np[te] == c] == c).sum(), (y_np[te] == c).sum()]
            accs.append(right / len(y_np))
        print(f"  {name:7} {np.mean(accs):6.1%} ({min(accs):.1%}-{max(accs):.1%})   "
              + "  ".join(f"{l} {per[c, 0] / per[c, 1]:.0%}" for c, l in enumerate(labels)), flush=True)


if __name__ == "__main__":
    main()
