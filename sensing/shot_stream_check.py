"""
shot_stream_check.py - does SHOT-IM survive a realistic, mostly-empty stream?

    python shot_stream_check.py <dataset.npz> --train 1 2 3 4 --streams <cued session folder> [...] [--calib 7 8]

Learns the CNN on the --train sessions (5 seeds), then for each cued session (windows every 0.5 s):
  source          no change
  AdaBN all       streaming: batch-norm averages re-measured on every window seen so far
  AdaBN blend     streaming: half learned averages, half those of every window seen so far
  SHOT-IM stream  SHOT-IM run once on all of the stream's windows, then scored on the same windows
With --calib (sessions of balanced clips from the stream's room, labels never used):
  AdaBN calib     AdaBN on the calibration clips, then left unchanged for the stream
  SHOT-IM calib   SHOT-IM on the calibration clips, then left unchanged for the stream
No method sees a label; labels are used only for scoring. SHOT-IM settings are adapt.py's, fixed in advance.
"""
import argparse
import numpy as np
import torch
from train_cnn import fit, prepare
from adapt import adabn, shot_im, predict_proba
from stream_check import run_stream, SEEDS
from norm_check import windows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--train", type=int, nargs="+", required=True)
    ap.add_argument("--calib", type=int, nargs="*", default=[])
    ap.add_argument("--streams", nargs="+", required=True)
    a = ap.parse_args()
    d = np.load(a.dataset)
    labels = [str(l) for l in d["labels"]]
    k = len(labels)
    P, y, ses = prepare(d["X"]), torch.from_numpy(d["y"].astype(np.int64)), d["session"]
    S = sorted(set(ses))
    pick = lambda nums: np.where(np.isin(ses, [S[i - 1] for i in nums]))[0]
    tr, ca = pick(a.train), pick(a.calib)
    print(f"learn on sessions {a.train} ({len(tr)} clips)" + (f"; calibration clips: sessions {a.calib} ({len(ca)} clips, labels not used)" if len(ca) else ""))
    srcs = [fit(P[tr], y[tr], k, seed) for seed in range(SEEDS)]
    calib = {}
    if len(ca):
        calib["AdaBN calib"] = [adabn(m, sc, P[ca]) for m, sc in srcs]
        calib["SHOT-IM calib"] = [shot_im(m, sc, P[ca], seed) for seed, (m, sc) in enumerate(srcs)]
    print("models ready", flush=True)

    for sd in a.streams:
        Xs, ys = windows(sd, labels)
        name = sd.rstrip("/").split("session_")[-1]
        print(f"\n[{name}] {len(ys)} windows; mix: " + "  ".join(f"{l} {np.mean(ys == c):.0%}" for c, l in enumerate(labels)))
        print(f"  {'method':15} {'accuracy':>8} {'range over seeds':>17} {'per-activity avg':>17}   per activity {'':30} share of windows given each label")
        methods = ["source", "AdaBN all", "AdaBN blend", "SHOT-IM stream"] + list(calib)
        for meth in methods:
            acc, bal, per, share = [], [], np.zeros((k, 2)), np.zeros(k)
            for seed, (src, scale) in enumerate(srcs):
                if meth == "SHOT-IM stream":
                    p = predict_proba(shot_im(src, scale, Xs, seed), scale, Xs).argmax(1).numpy()
                elif meth in calib:
                    p = predict_proba(calib[meth][seed], scale, Xs).argmax(1).numpy()
                else:
                    p = run_stream(src, scale, Xs, meth, k, seed)
                acc.append((p == ys).mean())                                   # labels used only from here on
                bal.append(np.mean([(p[ys == c] == c).mean() for c in range(k) if (ys == c).any()]))
                share += np.bincount(p, minlength=k) / len(p) / SEEDS
                for c in range(k):
                    per[c] += [(p[ys == c] == c).sum(), (ys == c).sum()]
            print(f"  {meth:15} {np.mean(acc):8.1%} {min(acc):8.1%}-{max(acc):.1%} {np.mean(bal):17.1%}   "
                  + "  ".join(f"{l} {per[c, 0] / per[c, 1]:4.0%}" for c, l in enumerate(labels) if per[c, 1])
                  + "     " + "  ".join(f"{l} {share[c]:.0%}" for c, l in enumerate(labels)), flush=True)


if __name__ == "__main__":
    main()
