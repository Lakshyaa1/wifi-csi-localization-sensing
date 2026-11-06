"""
sit_check.py - why are sit clips called empty: is their signal weak, or does the model miss them?

    python sit_check.py dataset.npz

Gets every clip's prediction from models that never saw its session (the same test as train_cnn.py),
then compares how strongly each clip's signal moves.
Strength: relative change per subcarrier, spread over a sliding 1 s window, averaged over
subcarriers, peak over time, stronger of the two boards. Peak suits sit, a single short event.
"""
import argparse
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from train_baseline import features
from train_cnn import fit, predict, prepare


def strength(clip, fs):
    """Returns (peak strength, median strength, time of peak in s) for one clip."""
    peaks, medians, times = [], [], []
    w = fs
    for A in clip:
        R = A / (A.mean(axis=-1, keepdims=True) + 1e-6)
        win = np.lib.stride_tricks.sliding_window_view(R, w, axis=-1)
        s = win.std(axis=-1).mean(axis=0)
        peaks.append(s.max()); medians.append(np.median(s)); times.append((s.argmax() + w / 2) / fs)
    b = int(np.argmax(peaks))
    return peaks[b], medians[b], times[b]


def describe(v):
    return f"median {np.median(v):.4f}  (middle half {np.percentile(v, 25):.4f}-{np.percentile(v, 75):.4f})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    d = np.load(a.dataset)
    raw, y_np, session, fs = d["X"], d["y"], d["session"], int(d["fs"])
    labels = [str(l) for l in d["labels"]]
    SIT, EMPTY = labels.index("sit"), labels.index("empty")
    S = sorted(set(session))
    k = len(labels)
    X, y = prepare(raw), torch.from_numpy(y_np.astype(np.int64))

    votes = np.zeros((len(y), k), int)
    for seed in range(a.seeds):
        for s in S:
            tr, te = np.where(session != s)[0], np.where(session == s)[0]
            m, sc = fit(X[tr], y[tr], k, seed)
            votes[te, predict(m, sc, X[te])] += 1
    cnn = votes.argmax(1)

    F = np.stack([features(c, fs) for c in raw])
    lr = np.zeros(len(y), int)
    for s in S:
        tr, te = session != s, session == s
        lr[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(F[tr], y_np[tr]).predict(F[te])

    st = np.array([strength(c, fs) for c in raw])
    peak, med, when = st[:, 0], st[:, 1], st[:, 2]
    sit, empty = y_np == SIT, y_np == EMPTY

    print("Does logistic regression also miss sit? (leave-one-session-out, same clips)")
    for i, l in enumerate(labels):
        m = y_np == i
        print(f"  {l:6} right {(lr[m] == i).sum()}/{m.sum()}, called empty {(lr[m] == EMPTY).sum()}")
    print(f"  all {(lr == y_np).sum()}/{len(y)}")

    print("\nCNN majority vote over seeds, sit clips:")
    for i, l in enumerate(labels):
        print(f"  called {l:6} {(cnn[sit] == i).sum()}")

    print("\nPeak strength by group")
    groups = [("empty, all", empty), ("empty, called empty", empty & (cnn == EMPTY)),
              ("sit, called empty", sit & (cnn == EMPTY)), ("sit, called sit", sit & (cnn == SIT)),
              ("sit, called other", sit & (cnn != EMPTY) & (cnn != SIT))]
    for name, m in groups:
        if m.sum():
            print(f"  {name:22} n={m.sum():3}  {describe(peak[m])}")

    print("\nBy session (median peak; sit vs empty separation, 0.5 = none, 1.0 = perfect)")
    for i, s in enumerate(S):
        q = session == s
        print(f"  S{i + 1}: empty {np.median(peak[q & empty]):.4f}  sit {np.median(peak[q & sit]):.4f}  "
              f"separation {roc_auc_score(sit[q & (sit | empty)], peak[q & (sit | empty)]):.2f}  "
              f"sit right (CNN) {(cnn[q & sit] == SIT).sum()}/{(q & sit).sum()}, "
              f"called empty {(cnn[q & sit] == EMPTY).sum()}")
    both = sit | empty
    print(f"  all sessions together: separation {roc_auc_score(sit[both], peak[both]):.2f}")

    print("\nSit clips vs the same session's empty clips")
    above = 0
    for i in np.where(sit)[0]:
        q = (session == session[i]) & empty
        above += peak[i] > np.percentile(peak[q], 90)
    print(f"  stronger than 90% of their session's empty clips: {above}/{sit.sum()}")
    for name, m in (("called empty", sit & (cnn == EMPTY)), ("called sit", sit & (cnn == SIT))):
        n = sum(peak[i] > np.percentile(peak[(session == session[i]) & empty], 90) for i in np.where(m)[0])
        print(f"    {name}: {n}/{m.sum()}")

    print("\nHow much of the clip is motion: peak / median strength")
    for name, m in (("empty", empty), ("sit, called empty", sit & (cnn == EMPTY)), ("sit, called sit", sit & (cnn == SIT))):
        if m.sum():
            print(f"  {name:20} median {np.median(peak[m] / med[m]):.2f}")

    print("\nTime of peak in sit clips, seconds into the 5 s clip")
    for name, m in (("called empty", sit & (cnn == EMPTY)), ("called sit", sit & (cnn == SIT))):
        if m.sum():
            print(f"  {name:13} " + " ".join(f"{t:.1f}" for t in sorted(when[m])))


if __name__ == "__main__":
    main()
