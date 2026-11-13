"""
live_extras.py - two ideas for the live classifier, scored on the scripted live-test session.

    python live_extras.py <live test session folder> <dataset.npz>

Windows: 5 s every 0.5 s inside the blocks of known actions (as score_live_test.py); models learn on every
clip of dataset.npz (all lower-sra-room sessions).
  A. logistic regression: 16 usual features vs 16 + 4 phase-difference features vs 4 phase-difference only
  B. CNN (5 seeds), in time order, as a live system would run:
       source:       no adaptation
       causal AdaBN: batch-norm averages re-measured on every window seen so far (all blocks, including
                     'seated', no labels), refreshed every 10 windows (5 s); the window being scored is used
                     only if already seen; the first refresh uses the first 10 windows
       full AdaBN:   re-measured once on all windows of the session (not possible live; upper reference)
Labels are read only for scoring.
"""
import csv, sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from csi_common import load_raw_log, process_clip
from build_dataset import DATA_SUBCARRIERS
from phase_check import build, clip_phase
from train_baseline import features
from train_cnn import fit, prepare
from adapt import adabn, predict_proba

FS, N, SEEDS, REFRESH = 100, 500, 5, 10


def main():
    sdir, dataset = sys.argv[1], sys.argv[2]
    root = sdir.rstrip("/").rsplit("/", 1)[0]
    d, P, _, _ = build(dataset, root)
    labels = [str(l) for l in d["labels"]]
    k = len(labels)

    boards = [load_raw_log(f"{sdir}/raw_board{i}.log") for i in (0, 1)]
    rows = list(csv.DictReader(open(f"{sdir}/index.csv", newline="")))
    wins, wph, wlab, wt = [], [], [], []           # all windows in time order; label -1 = unknown action
    for r in rows:
        a, b, lab = float(r["t_start"]), float(r["t_end"]), r["label"].strip()
        for t1 in np.arange(a + 5, b + 1e-6, 0.5):
            amps, phs = [], []
            for bd in boards:
                A, info = process_clip(bd, t1 - 5, t1, DATA_SUBCARRIERS, fs=FS, n=N)
                if A is None or not info["ok"]:
                    break
                p, _, _ = clip_phase(bd, t1 - 5, t1, N)
                amps.append(A.T); phs.append(p.T)
            if len(amps) < 2:
                continue
            wins.append(np.stack(amps)); wph.append(np.stack(phs))
            wlab.append(labels.index(lab) if lab in labels else -1); wt.append(t1)
    W, WP, yl = np.stack(wins).astype(np.float32), np.stack(wph), np.array(wlab)
    known = yl >= 0
    print(f"live windows: {len(W)} in time order, {known.sum()} of known actions")

    def dfeat(Pc):
        D = np.abs(np.angle(np.exp(1j * np.diff(Pc, axis=-1))))
        return np.c_[D.mean(axis=(2, 3)), np.percentile(D.reshape(len(D), 2, -1), 90, axis=-1)]

    F, Fl = np.stack([features(c, FS) for c in d["X"]]), np.stack([features(c, FS) for c in W])
    Fd, Fdl = dfeat(P), dfeat(WP)
    print("\nA. logistic regression, learned on all 152 clips")
    for name, (Xtr, Xte) in {"16 usual features": (F, Fl), "16 + 4 phase-difference": (np.c_[F, Fd], np.c_[Fl, Fdl]),
                             "4 phase-difference only": (Fd, Fdl)}.items():
        p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, d["y"]).predict(Xte)
        yt, pt = yl[known], p[known]
        print(f"  {name:26} {(pt == yt).mean():.1%}   " + "  ".join(
            f"{l} {(pt[yt == c] == c).mean():.0%}" for c, l in enumerate(labels)))

    print(f"\nB. CNN, {SEEDS} seeds")
    X, y = prepare(d["X"]), torch.from_numpy(d["y"].astype(np.int64))
    Xl = prepare(W)
    acc = {"source": [], "causal AdaBN": [], "full AdaBN": []}
    per = {m: np.zeros((k, 2)) for m in acc}
    for seed in range(SEEDS):
        src, scale = fit(X, y, k, seed)
        preds = {"source": predict_proba(src, scale, Xl).argmax(1).numpy(),
                 "full AdaBN": predict_proba(adabn(src, scale, Xl), scale, Xl).argmax(1).numpy()}
        causal = np.zeros(len(Xl), int)
        for start in range(0, len(Xl), REFRESH):
            seen = max(start, REFRESH)
            m = adabn(src, scale, Xl[:seen])
            causal[start:start + REFRESH] = predict_proba(m, scale, Xl[start:start + REFRESH]).argmax(1).numpy()
        preds["causal AdaBN"] = causal
        for name, p in preds.items():
            acc[name].append((p[known] == yl[known]).mean())
            for c in range(k):
                per[name][c] += [(p[known][yl[known] == c] == c).sum(), (yl[known] == c).sum()]
    for name, v in acc.items():
        print(f"  {name:13} mean {np.mean(v):.1%}, per seed " + " ".join(f"{a:.1%}" for a in v) + "   "
              + "  ".join(f"{l} {per[name][c, 0] / per[name][c, 1]:.0%}" for c, l in enumerate(labels)))


if __name__ == "__main__":
    main()
