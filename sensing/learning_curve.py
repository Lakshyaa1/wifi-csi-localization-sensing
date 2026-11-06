"""
learning_curve.py - would more clips help, or is the limit the change between sessions?

    python learning_curve.py dataset.npz

Uses the same features and classifier as train_baseline.py. Test clips are never used for
learning. Training clips are drawn at random, with classes balanced, many times over.
  1. Learning curve: test each session after learning on 10, 20, 40 or all clips of the others.
  2. Same number of clips from one other session vs a mix of two other sessions.
  3. Same test clips, after learning on clips from their own session vs another session.
"""
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from train_baseline import features
from eval_sessions import rescale_per_session


def accuracy(F, y, train, test):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return (clf.fit(F[train], y[train]).predict(F[test]) == y[test]).mean()


def quotas(n, classes, rng):
    """Split n clips across classes as evenly as possible; leftovers go to random classes."""
    q = {c: n // len(classes) for c in classes}
    for c in rng.choice(classes, n % len(classes), replace=False):
        q[c] += 1
    return q


def draw(pool, y, n, rng):
    """n clips from pool with classes balanced."""
    classes = np.unique(y)
    q = quotas(n, classes, rng)
    return np.concatenate([rng.choice(pool[y[pool] == c], q[c], replace=False) for c in classes])


def draw_mix(pool_a, pool_b, y, n, rng):
    """n clips with classes balanced, each class split evenly between two pools."""
    out = []
    classes = np.unique(y)
    for c, k in quotas(n, classes, rng).items():
        a, b = pool_a[y[pool_a] == c], pool_b[y[pool_b] == c]
        ka = k // 2 + (k % 2 if rng.random() < 0.5 else 0)
        out += [rng.choice(a, ka, replace=False), rng.choice(b, k - ka, replace=False)]
    return np.concatenate(out)


def summary(values, n_test):
    v = np.asarray(values)
    return f"{v.mean():6.1%}  (middle 80% of draws {np.percentile(v, 10):.1%}-{np.percentile(v, 90):.1%}, {n_test} test clips)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = np.load(a.dataset)
    X, y, session, fs = d["X"], d["y"], d["session"], int(d["fs"])
    labels = [str(l) for l in d["labels"]]
    F = np.stack([features(c, fs) for c in X])
    G = rescale_per_session(F, session)
    S = sorted(set(session))
    idx = {s: np.where(session == s)[0] for s in S}
    name = {s: f"S{i + 1}" for i, s in enumerate(S)}
    rng = np.random.default_rng(a.seed)

    print(f"{len(y)} clips, chance {1 / len(labels):.0%}")
    for s in S:
        print(f"  {name[s]}:", {l: int((y[idx[s]] == i).sum()) for i, l in enumerate(labels)})

    for title, Z in (("1. Learning curve, plain features", F),
                     ("1b. Learning curve, per-session rescaling (no labels)", G)):
        print(f"\n{title}")
        for n in (10, 20, 40, None):
            per_test, hits = [], 0
            for s in S:
                pool = np.concatenate([idx[o] for o in S if o != s])
                if n is None:
                    acc = [accuracy(Z, y, pool, idx[s])]
                else:
                    acc = [accuracy(Z, y, draw(pool, y, n, rng), idx[s]) for _ in range(a.draws)]
                per_test.append(np.mean(acc))
                hits += np.mean(acc) * len(idx[s])
            tag = "all (73-80)" if n is None else f"{n}"
            print(f"  {tag:>12} clips: " + "  ".join(f"{name[s]} {p:.1%}" for s, p in zip(S, per_test))
                  + f"  | all test clips {hits / len(y):.1%}")

    print("\n2. 20 training clips (5 per class): one other session vs a mix of the two others")
    for s in S:
        others = [o for o in S if o != s]
        for o in others:
            acc = [accuracy(F, y, draw(idx[o], y, 20, rng), idx[s]) for _ in range(a.draws)]
            print(f"  test {name[s]}, from {name[o]} only:  {summary(acc, len(idx[s]))}")
        acc = [accuracy(F, y, draw_mix(idx[others[0]], idx[others[1]], y, 20, rng), idx[s])
               for _ in range(a.draws)]
        print(f"  test {name[s]}, mix of both:   {summary(acc, len(idx[s]))}")

    print("\n3. 16 training clips (4 per class): same test clips, learned from own vs another session")
    all_same, all_other = [], []
    for s in S:
        others = [o for o in S if o != s]
        same, other = [], []
        for _ in range(a.draws):
            tr = draw(idx[s], y, 16, rng)
            te = np.setdiff1d(idx[s], tr)
            same.append(accuracy(F, y, tr, te))
            o = others[rng.integers(2)]
            other.append(accuracy(F, y, draw(idx[o], y, 16, rng), te))
        all_same += same
        all_other += other
        print(f"  test {name[s]}: own session {np.mean(same):.1%}, another session {np.mean(other):.1%}"
              f"  ({len(idx[s]) - 16} test clips each draw)")
    print(f"  average: own session {np.mean(all_same):.1%}, another session {np.mean(all_other):.1%}")


if __name__ == "__main__":
    main()
