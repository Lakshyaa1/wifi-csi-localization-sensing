"""
phase_check.py - does cleaned CSI phase help, next to amplitude?

    python phase_check.py <dataset.npz> <data folder with the session folders>

For exactly the clips in dataset.npz, the phase is rebuilt from the raw logs:
  1. per packet: angle of the 52 data subcarriers, unwrapped across subcarriers, then the best straight
     line over subcarrier index k is subtracted (removes the random per-packet slope and offset that come
     from the unsynchronised clocks)
  2. same packets dropped as for amplitude (gain glitches), resampled to 100 Hz, Hampel, 20 Hz low-pass
  3. per clip: minus each subcarrier's average over the clip (change over time, like the amplitude's
     relative change), then 50 Hz
Phase channels are scaled to the same overall spread as the amplitude channels (one number from all clips,
no labels). Compared with the CNN settings of train_cnn.py, 5 seeds:
  amplitude only (2 channels) / phase only (2) / amplitude + phase (4)
  tests: leave one session out, and sessions 1-3 -> 4.
Also printed first: a check that the rebuilt amplitude matches the dataset, and how much the phase moves
for each action (median over clips of the spread over time), to see whether phase follows motion at all.
"""
import csv, os, sys, time
import numpy as np
import torch
from csi_common import load_raw_log, to_complex, resample, hampel, lowpass, gain_outliers
from build_dataset import DATA_SUBCARRIERS
from train_cnn import fit, predict, prepare

FS, SEEDS = 100, 5
K = np.r_[-26:0, 1:27].astype(float)          # subcarrier index of each kept column


def clean_phase(H):
    """H: packets x 52 complex -> residual phase after removing a straight line over k, per packet."""
    H = H.copy()
    bad = ~np.isfinite(H)
    if bad.any():                              # first-word-invalid packets: borrow the next subcarrier
        for j in np.where(bad.any(0))[0]:
            nb = j + 1 if j + 1 < H.shape[1] else j - 1
            H[bad[:, j], j] = H[bad[:, j], nb]
    ph = np.unwrap(np.angle(H), axis=1)
    A = np.c_[K, np.ones_like(K)]
    coef, *_ = np.linalg.lstsq(A, ph.T, rcond=None)     # (2, packets)
    res = ph - (A @ coef).T
    res[bad] = np.nan
    return res, coef[0]


def clip_phase(board, t0, t1, n):
    t = board["board_t"]
    sel = (t >= t0 - 0.5) & (t <= t1 + 0.5)
    H = to_complex(board["csi"][sel], board["first_word_invalid"][sel])[:, DATA_SUBCARRIERS]
    glitch = gain_outliers(np.abs(H))
    res, slope = clean_phase(H[~glitch])
    _, P = resample(t[sel][~glitch], res, t0, n, FS)
    P, _ = hampel(P)
    P = lowpass(P, FS)
    _, A = resample(t[sel][~glitch], np.abs(H[~glitch]), t0, n, FS)
    A, _ = hampel(A)
    A = lowpass(A, FS)
    return P, A, slope


def build(dataset, root):
    d = np.load(dataset)
    n = d["X"].shape[-1]
    P = np.zeros(d["X"].shape, np.float32)
    amp_err, slopes = [], []
    cache = {}
    for i, (s, cid) in enumerate(zip(d["session"], d["clip_id"])):
        if s not in cache:
            sdir = os.path.join(root, str(s))
            rows = {int(r["clip_id"]): r for r in csv.DictReader(open(os.path.join(sdir, "index.csv")))}
            boards = [load_raw_log(os.path.join(sdir, f"raw_board{b}.log")) for b in (0, 1)]
            cache = {s: (rows, boards)}
        rows, boards = cache[s]
        t0 = float(rows[int(cid)]["t_start"])
        for b in range(2):
            p, a, sl = clip_phase(boards[b], t0, t0 + n / FS, n)
            P[i, b] = p.T
            slopes.append(np.median(np.abs(sl)))
            amp_err.append(np.abs(a.T - d["X"][i, b]).max() / (np.abs(d["X"][i, b]).mean() + 1e-9))
    return d, P, np.array(amp_err), np.array(slopes)


def main():
    dataset, root = sys.argv[1], sys.argv[2]
    t0 = time.time()
    d, P, amp_err, slopes = build(dataset, root)
    labels = [str(l) for l in d["labels"]]
    y_np, session = d["y"], d["session"]
    print(f"{len(y_np)} clips rebuilt in {time.time() - t0:.0f} s; non-finite phase values {int((~np.isfinite(P)).sum())}")
    print(f"amplitude rebuilt vs dataset: max relative difference median {np.median(amp_err):.2e}, worst {amp_err.max():.2e}")
    print(f"per-packet line slope removed: median {np.median(slopes):.2f} rad per subcarrier")

    Pc = P - P.mean(axis=-1, keepdims=True)                  # change over the clip
    move = Pc.std(axis=-1).mean(axis=-1)                      # (clips, boards): spread over time, avg over subcarriers
    Xa = prepare(d["X"])
    amove = Xa.numpy().std(axis=-1).mean(axis=-1)
    print("\nhow much each signal moves, median over clips (board 0 / board 1)")
    print(f"  {'action':6}  {'phase (rad)':>16}  {'amplitude (rel)':>18}")
    for c, l in enumerate(labels):
        m, am = np.median(move[y_np == c], 0), np.median(amove[y_np == c], 0)
        print(f"  {l:6}  {m[0]:7.3f} / {m[1]:6.3f}  {am[0]:8.3f} / {am[1]:7.3f}")

    Xp = torch.from_numpy(Pc[..., ::2].copy())
    Xp = Xp * (Xa.std() / Xp.std())                          # same overall spread as amplitude
    inputs = {"amplitude": Xa, "phase": Xp, "amplitude + phase": torch.cat([Xa, Xp], 1)}
    y = torch.from_numpy(y_np.astype(np.int64))
    k = len(labels)
    S = sorted(set(session))
    splits = {"leave one session out": [(np.where(session != s)[0], np.where(session == s)[0]) for s in S],
              "sessions 1-3 -> 4": [(np.where(np.isin(session, S[:3]))[0], np.where(session == S[3])[0])]}
    for split, folds in splits.items():
        print(f"\n{split}")
        for name, X in inputs.items():
            t1 = time.time()
            accs, right = [], np.zeros(k, int)
            for seed in range(SEEDS):
                hits = cnt = 0
                for tr, te in folds:
                    m, sc = fit(X[tr], y[tr], k, seed)
                    p = predict(m, sc, X[te])
                    hits += (p == y_np[te]).sum(); cnt += len(te)
                    right += [(p[y_np[te] == c] == c).sum() for c in range(k)]
                accs.append(hits / cnt)
            tot = sum(np.bincount(y_np[te], minlength=k) for _, te in folds) * SEEDS
            print(f"  {name:18} mean {np.mean(accs):.1%}, std {np.std(accs):.1%}, per seed "
                  + " ".join(f"{v:.1%}" for v in accs) + "   " + "  ".join(
                      f"{l} {r}/{c}" for l, r, c in zip(labels, right, tot)) + f"  ({time.time() - t1:.0f} s)",
                  flush=True)


if __name__ == "__main__":
    main()
