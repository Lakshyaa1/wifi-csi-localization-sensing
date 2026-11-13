"""
strength_check.py - does training with random strength scaling make the CNN less sensitive to session drift?

    python strength_check.py dataset.npz

Each training clip's signal is multiplied, per board, by a random factor between 0.5x and 2x (even on a
log scale, fresh every epoch), so "louder" alone can't decide the action. Compared with the plain CNN:
  1. leave one session out: test each session after learning on all the others, 5 seeds
  2. the earlier split: learn on sessions 1-3, test on session 4
  3. shuffled labels with scaling on (should be near chance)
  4. sensitivity: how often the guess changes when a test clip is scaled 0.5x or 2x
All other settings are those of train_cnn.py. The range 0.5-2x is fixed in advance (the median motion level
moved by about 2x between session 4 and the live test). Scores are from the last epoch.
"""
import sys, time
import numpy as np
import torch
from train_cnn import fit, predict, prepare

AUG = (0.5, 2.0)
SEEDS = 5


def main():
    d = np.load(sys.argv[1])
    X, y_np, session = prepare(d["X"]), d["y"], d["session"]
    y = torch.from_numpy(y_np.astype(np.int64))
    labels = [str(l) for l in d["labels"]]
    k = len(labels)
    S = sorted(set(session))
    print(f"{len(y)} clips, {len(S)} sessions, labels {labels}, strength scaling {AUG[0]}-{AUG[1]}x")
    variants = {"plain": None, "scaled": AUG}

    print("\n1. Leave one session out")
    t0 = time.time()
    flip = {v: [] for v in variants}
    for v, aug in variants.items():
        per_seed, per_sess = [], {s: [] for s in S}
        right = np.zeros(k, int)
        for seed in range(SEEDS):
            hits = 0
            for s in S:
                tr, te = np.where(session != s)[0], np.where(session == s)[0]
                m, sc = fit(X[tr], y[tr], k, seed, strength_aug=aug)
                p = predict(m, sc, X[te])
                per_sess[s].append((p == y_np[te]).mean())
                hits += (p == y_np[te]).sum()
                right += [(p[y_np[te] == c] == c).sum() for c in range(k)]
                for f in (0.5, 2.0):
                    flip[v].append((predict(m, sc, X[te] * f) != p).mean())
            per_seed.append(hits / len(y))
        n = np.bincount(y_np, minlength=k) * SEEDS
        print(f"  {v:6}: mean {np.mean(per_seed):.1%}, std {np.std(per_seed):.1%}, per seed "
              + " ".join(f"{a:.1%}" for a in per_seed))
        print("          per session " + "  ".join(f"S{i + 1} {np.mean(per_sess[s]):.1%}" for i, s in enumerate(S))
              + "   per class " + "  ".join(f"{l} {r}/{c}" for l, r, c in zip(labels, right, n)))
    print(f"  ({time.time() - t0:.0f} s)")

    print("\n2. Learn on sessions 1-3, test on session 4")
    tr, te = np.where(np.isin(session, S[:3]))[0], np.where(session == S[3])[0]
    for v, aug in variants.items():
        accs = []
        for seed in range(SEEDS):
            m, sc = fit(X[tr], y[tr], k, seed, strength_aug=aug)
            accs.append((predict(m, sc, X[te]) == y_np[te]).mean())
        print(f"  {v:6}: mean {np.mean(accs):.1%}, per seed " + " ".join(f"{a:.1%}" for a in accs))

    print("\n3. Shuffled training labels, scaling on, leave one session out")
    res = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(1000 + seed)
        hits = 0
        for s in S:
            tr, te = np.where(session != s)[0], np.where(session == s)[0]
            m, sc = fit(X[tr], y[tr][rng.permutation(len(tr))], k, seed, strength_aug=AUG)
            hits += (predict(m, sc, X[te]) == y_np[te]).sum()
        res.append(hits / len(y))
    print("  per seed " + " ".join(f"{a:.1%}" for a in res) + f", mean {np.mean(res):.1%}")

    print("\n4. Share of test guesses that change when the test clip is scaled 0.5x or 2x (from part 1)")
    for v in variants:
        print(f"  {v:6}: {np.mean(flip[v]):.1%}")


if __name__ == "__main__":
    main()
