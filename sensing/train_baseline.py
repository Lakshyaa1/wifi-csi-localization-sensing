"""
train_baseline.py - the first experiment: does a model trained in one room still work in another?

    python train_baseline.py dataset.npz --source A --target B

Trains a simple classifier on hand-made motion features from the source room, then reports:
  1. accuracy on held-out source-room clips   (same room: should be high)
  2. accuracy on target-room clips            (new room: the domain-shift drop)
plus a confusion matrix for each, saved as baseline_confusion.png.

Features per clip and board: how much each subcarrier moves (std over time, summarised),
and how the strongest motion component's energy is spread over frequency bands.
This is deliberately simple. The CNN and the paper's unlabelled adaptation come next.
"""
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from csi_common import motion_component

BANDS = [(0.3, 2), (2, 5), (5, 10), (10, 20)]


def features(clip, fs):
    feats = []
    for A in clip:                                   # A: (subcarriers, time) for one board
        A = A.T
        A = A / (A.mean(axis=0, keepdims=True) + 1e-9)     # relative change, removes room-level scale
        s = A.std(axis=0)
        feats += [s.mean(), s.std(), np.percentile(s, 90)]
        m = motion_component(A)
        P = np.abs(np.fft.rfft(m * np.hanning(len(m)))) ** 2
        f = np.fft.rfftfreq(len(m), 1 / fs)
        tot = P[(f > 0.3) & (f < 20)].sum() + 1e-12
        feats += [np.log(tot)] + [P[(f >= lo) & (f < hi)].sum() / tot for lo, hi in BANDS]
    return np.array(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = np.load(a.dataset)
    X, y, room, fs = d["X"], d["y"], d["room"], int(d["fs"])
    labels = [str(l) for l in d["labels"]]
    F = np.stack([features(c, fs) for c in X])

    src, tgt = room == a.source, room == a.target
    Ftr, Fte, ytr, yte = train_test_split(F[src], y[src], test_size=0.3, stratify=y[src], random_state=a.seed)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(Ftr, ytr)

    acc_s = accuracy_score(yte, clf.predict(Fte))
    acc_t = accuracy_score(y[tgt], clf.predict(F[tgt]))
    chance = 1 / len(labels)
    print(f"classes: {labels}   (chance = {chance:.0%})")
    print(f"train: room {a.source}, {len(ytr)} clips")
    print(f"  same room  ({a.source}, held-out {len(yte)} clips): accuracy {acc_s:.1%}")
    print(f"  new room   ({a.target}, {tgt.sum()} clips):          accuracy {acc_t:.1%}")
    print(f"  drop: {100 * (acc_s - acc_t):.1f} points")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for k, (yt, Fx, ttl) in enumerate([(yte, Fte, f"same room {a.source}: {acc_s:.0%}"),
                                       (y[tgt], F[tgt], f"new room {a.target}: {acc_t:.0%}")]):
        cm = confusion_matrix(yt, clf.predict(Fx), labels=range(len(labels)), normalize="true")
        ax[k].imshow(cm, vmin=0, vmax=1, cmap="Blues")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax[k].text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                           color="white" if cm[i, j] > 0.5 else "black", fontsize=9)
        ax[k].set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
                  yticklabels=labels, xlabel="predicted", ylabel="true", title=ttl)
    fig.tight_layout()
    fig.savefig("baseline_confusion.png", dpi=110)
    print("saved baseline_confusion.png")


if __name__ == "__main__":
    main()
