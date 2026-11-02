"""
eval_sessions.py - how well does the baseline work on a session it has never seen?

    python eval_sessions.py dataset.npz

Uses the same features and classifier as train_baseline.py, but splits by session instead of room:
  1. learn on one or more whole sessions, test on another
  2. test each session after learning on all the others, with and without per-session rescaling
Per-session rescaling standardises each session's features with that session's own mean and spread.
It uses no labels. A shuffled-label run shows what chance looks like.
"""
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from train_baseline import features


def classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def rescale_per_session(F, session):
    G = F.copy()
    for s in np.unique(session):
        m = session == s
        G[m] = (F[m] - F[m].mean(axis=0)) / (F[m].std(axis=0) + 1e-9)
    return G


def report(name, F, y, train, test, labels):
    p = classifier().fit(F[train], y[train]).predict(F[test])
    print(f"{name}: {(p == y[test]).mean():.1%} ({(p == y[test]).sum()}/{test.sum()})")
    return confusion_matrix(y[test], p, labels=range(len(labels)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    a = ap.parse_args()
    d = np.load(a.dataset)
    X, y, session, fs = d["X"], d["y"], d["session"], int(d["fs"])
    labels = [str(l) for l in d["labels"]]
    F = np.stack([features(c, fs) for c in X])
    S = sorted(set(session))
    print(f"{len(y)} clips, {len(S)} sessions, classes {labels}, chance {1 / len(labels):.0%}")

    if len(S) >= 2:
        print("\nLearn on some sessions, test on another")
        total = report("  session 1 -> 2", F, y, session == S[0], session == S[1], labels)
        total = total + report("  session 2 -> 1", F, y, session == S[1], session == S[0], labels)
        if len(S) >= 3:
            total = total + report("  sessions 1+2 -> 3", F, y, np.isin(session, S[:2]), session == S[2], labels)
        print(f"  rows true, columns predicted {labels}")
        for l, row in zip(labels, total):
            print(f"  {l:6}", row.tolist())

    G = rescale_per_session(F, session)
    for name, Z in (("\nEach session tested after learning on the others", F),
                    ("\nSame, with per-session rescaling (no labels)", G)):
        print(name)
        for i, s in enumerate(S):
            report(f"  session {i + 1}", Z, y, session != s, session == s, labels)

    rng = np.random.default_rng(0)
    test = session == S[-1]
    acc = [(classifier().fit(F[~test], rng.permutation(y[~test])).predict(F[test]) == y[test]).mean()
           for _ in range(20)]
    print(f"\nShuffled labels, testing the last session (20 tries): {np.mean(acc):.1%}")


if __name__ == "__main__":
    main()
