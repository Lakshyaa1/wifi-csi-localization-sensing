"""
spec_check.py - does a ripple-rate picture (spectrogram) as the CNN input bring walk back?

    python spec_check.py <dataset.npz> <cued session folder> [...]

From the usual model input (relative change per subcarrier, 50 Hz, 5 s): for each board and subcarrier, the
power at each ripple rate 1..25 per second in 1 s slices taken every 0.1 s (Hann window), averaged over the
52 subcarriers, then log. Input to the same CNN: (2 boards, 25 rates, 41 time slices). Two versions:
  spec        log power with each clip's own average (per board) subtracted: a board that reads everything
              k times larger gives exactly the same picture (no reference level from other clips needed)
  spec-level  log power as it is (the level is kept)
Part 1: CNN learned on all training clips (5 seeds), each session replayed in order; no adaptation and
        AdaBN blend. Part 2: leave one training session out, no adaptation.
Compare with the 'none' rows of norm_check.py (same sessions, amplitude input). Settings fixed before running.
"""
import sys, time
import numpy as np
import torch
from train_cnn import fit, prepare
from adapt import predict_proba
from stream_check import run_stream, SEEDS
from norm_check import windows


def spec(P, level):
    n, b, c, t = P.shape
    S = torch.stft(P.reshape(-1, t), n_fft=50, hop_length=5, window=torch.hann_window(50), center=False,
                   return_complex=True).abs() ** 2                      # (n*b*c, 26, 41)
    S = S.reshape(n, b, c, 26, -1).mean(2)[:, :, 1:]                    # average subcarriers, drop rate 0
    L = torch.log(S + 1e-10)
    return L if level else L - L.mean(dim=(2, 3), keepdim=True)


def main():
    d = np.load(sys.argv[1])
    labels = [str(l) for l in d["labels"]]
    k = len(labels)
    P, y_np, ses = prepare(d["X"]), d["y"], d["session"]
    y = torch.from_numpy(y_np.astype(np.int64))
    S = sorted(set(ses))
    streams = {sd.rstrip("/").split("session_")[-1]: windows(sd, labels) for sd in sys.argv[2:]}
    kinds = {"spec": False, "spec-level": True}
    train = {name: spec(P, lv) for name, lv in kinds.items()}
    print("input shape", tuple(train["spec"].shape))

    print("Part 1 - learn on all training clips, replay each session in order (5 seeds)")
    t0 = time.time()
    models = {name: [fit(train[name], y, k, seed) for seed in range(SEEDS)] for name in kinds}
    print(f"  ({len(kinds) * SEEDS} models in {time.time() - t0:.0f} s)")
    for sname, (Xs, ys) in streams.items():
        known = ys >= 0
        print(f"\n[{sname}] {known.sum()} scored windows; mix: " + "  ".join(f"{l} {np.mean(ys[known] == c):.0%}" for c, l in enumerate(labels)))
        print(f"  {'input':10} {'method':12} {'right':>6} {'range over seeds':>17} {'per-activity avg':>17}   per activity")
        for name, lv in kinds.items():
            Xn = spec(Xs, lv)
            for meth in ("source", "AdaBN blend"):
                acc, bal, per = [], [], np.zeros((k, 2))
                for seed, (src, scale) in enumerate(models[name]):
                    p = run_stream(src, scale, Xn, meth, k, seed)
                    acc.append((p[known] == ys[known]).mean())
                    bal.append(np.mean([(p[ys == c] == c).mean() for c in range(k) if (ys == c).any()]))
                    for c in range(k):
                        per[c] += [(p[ys == c] == c).sum(), (ys == c).sum()]
                print(f"  {name:10} {meth:12} {np.mean(acc):6.1%} {min(acc):8.1%}-{max(acc):.1%} {np.mean(bal):17.1%}   "
                      + "  ".join(f"{l} {per[c, 0] / per[c, 1]:.0%}" for c, l in enumerate(labels) if per[c, 1]), flush=True)

    print("\nPart 2 - leave one training session out, no adaptation (5 seeds)")
    for name in kinds:
        accs, per = [], np.zeros((k, 2))
        for seed in range(SEEDS):
            right = 0
            for s_ in S:
                te, tr = ses == s_, ses != s_
                src, scale = fit(train[name][tr], y[tr], k, seed)
                p = predict_proba(src, scale, train[name][te]).argmax(1).numpy()
                right += (p == y_np[te]).sum()
                for c in range(k):
                    per[c] += [(p[y_np[te] == c] == c).sum(), (y_np[te] == c).sum()]
            accs.append(right / len(y_np))
        print(f"  {name:10} {np.mean(accs):6.1%} ({min(accs):.1%}-{max(accs):.1%})   "
              + "  ".join(f"{l} {per[c, 0] / per[c, 1]:.0%}" for c, l in enumerate(labels)), flush=True)


if __name__ == "__main__":
    main()
