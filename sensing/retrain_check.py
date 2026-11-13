"""
retrain_check.py - does adding the two 2026-11-12 training sessions (new board restarts, a third day) fix walk?

    python retrain_check.py <old dataset.npz> <new dataset.npz> <cued session folder> [...]

Training sets: old (sessions 1-4) and new (sessions 1-6). Inputs: amp (relative change, as always) and
spec (scale-free ripple-rate picture, spec_check.py). 5 seeds.
Part 1: learn on all clips of the training set, replay each cued session in order; no adaptation and AdaBN blend.
Part 2: new set only - leave one training session out, no adaptation.
Three of the cued sessions are from the same day as sessions 5-6 (different board restarts); 20261106 is not.
"""
import sys
import numpy as np
import torch
from train_cnn import fit, prepare
from adapt import predict_proba
from stream_check import run_stream, SEEDS
from norm_check import windows
from spec_check import spec

INPUTS = {"amp": lambda P: P, "spec": lambda P: spec(P, False)}


def main():
    part2 = "--no-part2" not in sys.argv
    sys.argv = [x for x in sys.argv if x != "--no-part2"]
    sets = {"S1-4": np.load(sys.argv[1]), "S1-6": np.load(sys.argv[2])}
    labels = [str(l) for l in sets["S1-4"]["labels"]]
    k = len(labels)
    streams = {sd.rstrip("/").split("session_")[-1]: windows(sd, labels) for sd in sys.argv[3:]}
    print("Part 1 - learn on all clips of the set, replay each cued session (5 seeds)")
    models = {}
    for sname, d in sets.items():
        P, y = prepare(d["X"]), torch.from_numpy(d["y"].astype(np.int64))
        for iname, tf in INPUTS.items():
            models[(sname, iname)] = [fit(tf(P), y, k, seed) for seed in range(SEEDS)]
        print(f"  {sname}: {len(y)} clips, models ready", flush=True)
    for stname, (Xs, ys) in streams.items():
        known = ys >= 0
        print(f"\n[{stname}] {known.sum()} scored windows; mix: " + "  ".join(f"{l} {np.mean(ys[known] == c):.0%}" for c, l in enumerate(labels)))
        print(f"  {'train':5} {'input':5} {'method':12} {'right':>6} {'range over seeds':>17} {'per-activity avg':>17}   per activity")
        for (sname, iname), ms in models.items():
            Xn = INPUTS[iname](Xs)
            for meth in ("source", "AdaBN blend"):
                acc, bal, per = [], [], np.zeros((k, 2))
                for seed, (src, scale) in enumerate(ms):
                    p = run_stream(src, scale, Xn, meth, k, seed)
                    acc.append((p[known] == ys[known]).mean())
                    bal.append(np.mean([(p[ys == c] == c).mean() for c in range(k) if (ys == c).any()]))
                    for c in range(k):
                        per[c] += [(p[ys == c] == c).sum(), (ys == c).sum()]
                print(f"  {sname:5} {iname:5} {meth:12} {np.mean(acc):6.1%} {min(acc):8.1%}-{max(acc):.1%} {np.mean(bal):17.1%}   "
                      + "  ".join(f"{l} {per[c, 0] / per[c, 1]:.0%}" for c, l in enumerate(labels) if per[c, 1]), flush=True)

    if not part2:
        return
    print("\nPart 2 - S1-6: leave one training session out, no adaptation (5 seeds)")
    d = sets["S1-6"]
    P, y_np, ses = prepare(d["X"]), d["y"], d["session"]
    y = torch.from_numpy(y_np.astype(np.int64))
    S = sorted(set(ses))
    for iname, tf in INPUTS.items():
        X = tf(P)
        accs, per, by = [], np.zeros((k, 2)), {s_: [] for s_ in S}
        for seed in range(SEEDS):
            right = 0
            for s_ in S:
                te, tr = ses == s_, ses != s_
                src, scale = fit(X[tr], y[tr], k, seed)
                p = predict_proba(src, scale, X[te]).argmax(1).numpy()
                right += (p == y_np[te]).sum(); by[s_].append((p == y_np[te]).mean())
                for c in range(k):
                    per[c] += [(p[y_np[te] == c] == c).sum(), (y_np[te] == c).sum()]
            accs.append(right / len(y_np))
        print(f"  {iname:5} {np.mean(accs):6.1%} ({min(accs):.1%}-{max(accs):.1%})   "
              + "  ".join(f"{l} {per[c, 0] / per[c, 1]:.0%}" for c, l in enumerate(labels))
              + "   by session: " + " ".join(f"{np.mean(v):.0%}" for v in by.values()), flush=True)


if __name__ == "__main__":
    main()
