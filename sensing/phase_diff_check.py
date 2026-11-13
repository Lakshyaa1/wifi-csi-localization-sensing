"""
phase_diff_check.py - is the phase *change between packets* (phase difference over time) a useful parameter?

    python phase_diff_check.py <dataset.npz> <data folder with the session folders>

Uses the cleaned phase of phase_check.py (per packet: unwrapped over subcarriers, straight line removed;
then 100 Hz, Hampel, 20 Hz low-pass). The phase difference is how much that phase moves from one 10 ms
sample to the next, wrapped to -pi..pi. Faster motion should change path lengths faster, so it should
change the phase faster.
  1. per action: median over clips of the average |phase difference| (rad per 10 ms), per board
  2. logistic regression: the 16 usual features vs 16 + 4 phase-difference features
     (per board: mean and 90th percentile of |phase difference|)
  3. CNN (train_cnn.py settings): amplitude vs phase difference only vs amplitude + phase difference
Tests: leave one session out, and sessions 1-3 -> 4; CNN over 5 seeds (logistic regression has no seed).
"""
import sys, time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from phase_check import build
from train_baseline import features
from train_cnn import fit, predict, prepare

SEEDS = 5


def main():
    dataset, root = sys.argv[1], sys.argv[2]
    t0 = time.time()
    d, P, _, _ = build(dataset, root)
    labels = [str(l) for l in d["labels"]]
    y_np, session = d["y"], d["session"]
    k = len(labels)
    D = np.angle(np.exp(1j * np.diff(P, axis=-1))).astype(np.float32)      # (clips, boards, 52, 499)
    absD = np.abs(D)
    print(f"{len(y_np)} clips rebuilt in {time.time() - t0:.0f} s")

    m = absD.mean(axis=(2, 3))
    print("\n1. average |phase difference| per 10 ms, median over clips (board 0 / board 1), radians")
    for c, l in enumerate(labels):
        v = np.median(m[y_np == c], 0)
        print(f"  {l:6} {v[0]:.4f} / {v[1]:.4f}")

    F = np.stack([features(c, 100) for c in d["X"]])
    Fd = np.c_[absD.mean(axis=(2, 3)), np.percentile(absD.reshape(len(D), 2, -1), 90, axis=-1)]
    S = sorted(set(session))
    splits = {"leave one session out": [(np.where(session != s)[0], np.where(session == s)[0]) for s in S],
              "sessions 1-3 -> 4": [(np.where(np.isin(session, S[:3]))[0], np.where(session == S[3])[0])]}

    print("\n2. logistic regression")
    for split, folds in splits.items():
        for name, X in {"16 usual features": F, "16 + 4 phase-difference": np.c_[F, Fd],
                        "4 phase-difference only": Fd}.items():
            hits = cnt = 0
            right = np.zeros(k, int)
            for tr, te in folds:
                p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y_np[tr]).predict(X[te])
                hits += (p == y_np[te]).sum(); cnt += len(te)
                right += [(p[y_np[te] == c] == c).sum() for c in range(k)]
            tot = sum(np.bincount(y_np[te], minlength=k) for _, te in folds)
            print(f"  {split:22} {name:26} {hits / cnt:.1%}   " + "  ".join(
                f"{l} {r}/{c}" for l, r, c in zip(labels, right, tot)))

    print("\n3. CNN")
    Xa = prepare(d["X"])
    Xd = torch.from_numpy(D[..., ::2].copy())
    Xd = Xd * (Xa.std() / Xd.std())
    y = torch.from_numpy(y_np.astype(np.int64))
    inputs = {"amplitude": Xa, "phase difference": Xd, "amplitude + phase diff": torch.cat([Xa, Xd], 1)}
    for split, folds in splits.items():
        print(f"  {split}")
        for name, X in inputs.items():
            accs, right = [], np.zeros(k, int)
            for seed in range(SEEDS):
                hits = cnt = 0
                for tr, te in folds:
                    mdl, sc = fit(X[tr], y[tr], k, seed)
                    p = predict(mdl, sc, X[te])
                    hits += (p == y_np[te]).sum(); cnt += len(te)
                    right += [(p[y_np[te] == c] == c).sum() for c in range(k)]
                accs.append(hits / cnt)
            tot = sum(np.bincount(y_np[te], minlength=k) for _, te in folds) * SEEDS
            print(f"    {name:24} mean {np.mean(accs):.1%}, std {np.std(accs):.1%}, per seed "
                  + " ".join(f"{v:.1%}" for v in accs) + "   " + "  ".join(
                      f"{l} {r}/{c}" for l, r, c in zip(labels, right, tot)), flush=True)


if __name__ == "__main__":
    main()
