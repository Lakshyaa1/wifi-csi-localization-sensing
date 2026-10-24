"""Classical baselines (Random Forest, SVM) on handcrafted window features.

Usage:
  python train_classical.py --target activity --train s01_roomA s02_roomA --test s03_roomA
  python train_classical.py --target zone --train s01_roomA s02_roomA --test s03_roomA
  python train_classical.py --target activity --train first_motion      # no --test: see below
  python train_classical.py --target zone --links 0 ...                 # first link only

Train and test must be different sessions: neighbouring windows overlap by 50%,
so a random split would leak test samples into training. Without --test, each
labelled phase of the recording is split in time (first 60% train, one window
skipped, rest test); that is only a smoke test and gives optimistic numbers.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import window_features
from layout import ZONE_XY
from preprocess import HOP

ROOT = Path(__file__).parent


def load(names, target, links):
    X, y, t0 = [], [], []
    for name in names:
        d = np.load(ROOT / "data" / "processed" / f"{name}.npz")
        x = d["X"]
        if links:   # keep only these links (two channels each)
            x = x[:, [c for l in links for c in (2 * l, 2 * l + 1)]]
        X.append(x)
        y.append(d[target].astype(str))
        t0.append(d["t0"])
    return np.concatenate(X), np.concatenate(y), np.concatenate(t0)


def zone_error(true, pred):
    """Mean distance (m) between true and predicted zone centres, where both are real zones."""
    d = [np.hypot(*np.subtract(ZONE_XY[a], ZONE_XY[b])) for a, b in zip(true, pred)
         if a in ZONE_XY and b in ZONE_XY]
    return float(np.mean(d)) if d else float("nan")


def time_split(y, t0, frac=0.6):
    """Train/test masks: first `frac` of every phase trains, one window is skipped, rest tests."""
    cuts = np.r_[0, np.where((y[1:] != y[:-1]) | (np.diff(t0) > 1.5 * HOP))[0] + 1, len(y)]
    train = np.zeros(len(y), bool)
    test = np.zeros(len(y), bool)
    for a, b in zip(cuts[:-1], cuts[1:]):
        k = a + int(round(frac * (b - a)))
        train[a:k] = True
        test[k + 1:b] = True
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["activity", "zone"], default="activity")
    ap.add_argument("--links", type=int, nargs="*", default=[], help="link indices to use (default: all)")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X, y, t0 = load(args.train, args.target, args.links)
    F = window_features(X)
    if args.test:
        Xte, yte, _ = load(args.test, args.target, args.links)
        Ftr, ytr, Fte = F, y, window_features(Xte)
        split = "session"
    else:
        tr, te = time_split(y, t0)
        Ftr, ytr, Fte, yte = F[tr], y[tr], F[te], y[te]
        split = "within-recording (optimistic)"
    classes = sorted({str(c) for c in ytr} | {str(c) for c in yte})
    print(f"target={args.target}  links={args.links or 'all'}  split={split}  train={len(ytr)}  test={len(yte)}  features={F.shape[1]}")
    print(f"classes: {classes}")

    models = {
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=args.seed),
        "svm_rbf": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
    }
    majority = max(classes, key=lambda c: (ytr == c).sum())
    preds = {"majority_class": np.array([majority] * len(yte))}
    for name, model in models.items():
        preds[name] = model.fit(Ftr, ytr).predict(Fte)
        print(f"\n{name} confusion matrix (rows = true, columns = predicted: {classes})")
        print(confusion_matrix(yte, preds[name], labels=classes))
    # sanity check: with shuffled training labels the model must fall to chance
    rng = np.random.default_rng(args.seed)
    preds["random_forest_shuffled_labels"] = models["random_forest"].fit(Ftr, rng.permutation(ytr)).predict(Fte)

    rows = [(name, accuracy_score(yte, p), f1_score(yte, p, average="macro", zero_division=0),
             zone_error(yte, p) if args.target == "zone" else float("nan")) for name, p in preds.items()]
    print(f"\n{'model':32s} accuracy  macro-F1" + ("  zone error (m)" if args.target == "zone" else ""))
    for name, acc, f1, err in rows:
        print(f"{name:32s} {acc:8.3f}  {f1:8.3f}" + (f"  {err:8.2f}" if args.target == "zone" else ""))
    out = ROOT / "results" / "classical.csv"
    out.parent.mkdir(exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["target", "links", "split", "train", "test", "model", "accuracy", "macro_f1", "zone_error_m"])
        for name, acc, f1, err in rows:
            w.writerow([args.target, "+".join(map(str, args.links)) or "all", split, "+".join(args.train),
                        "+".join(args.test), name, f"{acc:.4f}", f"{f1:.4f}", f"{err:.3f}"])


if __name__ == "__main__":
    main()
