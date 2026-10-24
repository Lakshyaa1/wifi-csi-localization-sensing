"""
check_quality.py - is a recorded session usable? Run it right after recording.

    python check_quality.py data/session_20261024_153000_roomA

Prints packet rate, gaps, RSSI and per-clip packet counts for each board, and saves
quality.png in the session folder (rate over time, gap histogram, RSSI, amplitude heatmap
with the clips marked).
"""
import argparse, csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from csi_common import load_raw_log, to_complex, pick_subcarriers


def read_index(sdir):
    p = os.path.join(sdir, "index.csv")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    a = ap.parse_args()
    logs = sorted(glob.glob(os.path.join(a.session, "raw_board*.log")))
    clips = read_index(a.session)
    if not logs:
        raise SystemExit("no raw_board*.log files in " + a.session)

    fig, axes = plt.subplots(len(logs), 4, figsize=(18, 3.6 * len(logs)), squeeze=False)
    verdict = []
    for row, path in enumerate(logs):
        name = os.path.basename(path)[4:-4]
        d = load_raw_log(path)
        t = d["board_t"]; t_rel = t - t[0]
        dur = t[-1] - t[0]
        gaps = np.diff(t)
        rate = len(t) / dur
        amp = np.abs(to_complex(d["csi"], d["first_word_invalid"]))
        sub = pick_subcarriers(amp)
        print(f"\n== {name} ==")
        print(f"  packets {len(t)}  over {dur:.1f} s   mean rate {rate:.1f}/s")
        print(f"  gaps: median {1e3*np.median(gaps):.1f} ms, 99th pct {1e3*np.percentile(gaps, 99):.1f} ms, "
              f"max {1e3*gaps.max():.0f} ms,  gaps > 50 ms: {(gaps > 0.05).sum()}")
        print(f"  RSSI mean {d['rssi'].mean():.1f} dBm, std {d['rssi'].std():.1f}")
        print(f"  CSI length {d['csi'].shape[1]} values -> {amp.shape[1]} subcarriers, {len(sub)} carry signal")
        if d["dropped_other_length"]:
            print(f"  ignored {d['dropped_other_length']} packets with a different CSI length")
        low = []
        for c in clips:
            t0, t1 = float(c["t_start"]), float(c["t_end"])
            n = int(((t >= t0) & (t <= t1)).sum())
            if n < 0.8 * 100 * (t1 - t0):
                low.append((c["clip_id"], c["label"], n))
        print(f"  clips: {len(clips)}, with < 80% of expected packets: {len(low)}"
              + (f"  e.g. {low[:5]}" if low else ""))
        ok = rate > 80 and (gaps > 0.25).sum() < 5 and not low
        verdict.append((name, ok))

        # plots
        bins = np.arange(0, dur + 1, 1.0)
        axes[row, 0].plot(bins[:-1], np.histogram(t_rel, bins)[0], lw=1)
        axes[row, 0].axhline(100, color="gray", ls="--", lw=0.8)
        axes[row, 0].set(title=f"{name}: packets per second", xlabel="time (s)")
        axes[row, 1].hist(np.clip(gaps * 1e3, 0, 100), bins=50)
        axes[row, 1].set(title="time between packets (ms, capped at 100)", yscale="log")
        axes[row, 2].plot(t_rel, d["rssi"], ".", ms=1)
        axes[row, 2].set(title="RSSI (dBm)", xlabel="time (s)")
        show = slice(0, min(len(t), 6000))
        A = amp[show][:, sub]
        axes[row, 3].imshow(A.T, aspect="auto", origin="lower", cmap="viridis",
                            extent=[t_rel[show][0], t_rel[show][-1], 0, len(sub)],
                            vmin=np.percentile(A, 2), vmax=np.percentile(A, 98))
        for c in clips:
            s0, s1 = float(c["t_start"]) - t[0], float(c["t_end"]) - t[0]
            if s0 < t_rel[show][-1]:
                axes[row, 3].axvspan(s0, s1, color="w", alpha=0.15)
                axes[row, 3].text(s0, len(sub) * 0.92, c["label"], color="w", fontsize=7)
        axes[row, 3].set(title="amplitude (first 60 s), clips shaded", xlabel="time (s)", ylabel="subcarrier")
    fig.tight_layout()
    out = os.path.join(a.session, "quality.png")
    fig.savefig(out, dpi=110)
    print("\nsaved", out)
    print("\nVerdict: " + ", ".join(f"{n} {'OK' if ok else 'CHECK'}" for n, ok in verdict))


if __name__ == "__main__":
    main()
