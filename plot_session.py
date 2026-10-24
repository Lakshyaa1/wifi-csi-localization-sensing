"""Plot a multi-receiver recording made by logger.py (link / activity / zone columns).

Usage:
  python plot_session.py test_short2
Produces plots/<name>_session.png with, on a shared time axis:
  - one amplitude heatmap per receiver (time x subcarrier), each divided by that
    receiver's median so both share one colour scale
  - the motion score per receiver: per-subcarrier std over a 0.5 s sliding window,
    averaged over subcarriers
Phase boundaries are marked with the activity and zone; transitions are shaded.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).parent
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]     # one per receiver, fixed order
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
GRID_HZ = 30.0


def load(name):
    """link -> (host time s, amplitude [packets, active subcarriers]); plus phase spans."""
    per, phases = {}, []
    with open(ROOT / "data" / f"{name}.csv") as f:
        for r in csv.DictReader(f):
            v = np.array(r["csi"].split(), dtype=float)[:128]
            per.setdefault(r["link"], []).append((float(r["host_time"]), np.hypot(v[0::2], v[1::2])))
            phases.append((float(r["host_time"]), r["activity"], r["zone"]))
    t0 = min(p[0] for p in phases)
    phases.sort()
    spans, start = [], 0
    for i in range(1, len(phases) + 1):
        if i == len(phases) or phases[i][1:] != phases[start][1:]:
            spans.append((phases[start][0] - t0, phases[i - 1][0] - t0, *phases[start][1:]))
            start = i
    out = {}
    for link, rows in sorted(per.items()):
        t = np.array([x[0] for x in rows]) - t0
        a = np.array([x[1] for x in rows])
        out[link] = (t, a[:, a.std(axis=0) > 0])
    return out, spans


def main():
    name = sys.argv[1]
    links, spans = load(name)
    t_end = max(s[1] for s in spans)
    grid = np.arange(0, t_end, 1 / GRID_HZ)
    n = len(links)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, ax = plt.subplots(n + 1, 1, figsize=(12, 3.1 * n + 3.2), sharex=True, facecolor=SURFACE,
                           gridspec_kw={"height_ratios": [1.5] * n + [1.3]})
    scores = {}
    for i, (link, (t, a)) in enumerate(links.items()):
        rel = a / np.median(a)
        img = np.stack([np.interp(grid, t, rel[:, j]) for j in range(rel.shape[1])])
        im = ax[i].imshow(img, aspect="auto", origin="lower", cmap=BLUES, vmin=0, vmax=2.5,
                          extent=[0, t_end, 0, rel.shape[1]], interpolation="nearest")
        ax[i].set_ylabel(f"{link}\nsubcarrier")
        ax[i].set_facecolor(SURFACE)
        rate = len(t) / (t[-1] - t[0])
        win = max(3, int(round(0.5 * rate)))
        scores[link] = (t, np.array([a[max(0, k - win + 1): k + 1].std(axis=0).mean() for k in range(len(a))]))
    cb = fig.colorbar(im, cax=ax[0].inset_axes([1.01, -0.6, 0.012, 1.6]))
    cb.set_label("amplitude ÷ receiver median")
    cb.outline.set_visible(False)

    s_ax = ax[-1]
    s_ax.set_facecolor(SURFACE)
    for (link, (t, s)), colour in zip(scores.items(), SERIES):
        s_ax.plot(t, s, color=colour, lw=1.6, label=link)
    s_ax.set_ylabel("motion score\n(0.5 s std of amplitude)")
    s_ax.set_xlabel("time since start of recording (s)")
    # a short burst would flatten everything else: cap the axis and say where it was cut
    allscores = np.concatenate([s for _, s in scores.values()])
    top = 1.8 * np.percentile(allscores, 95)
    s_ax.set_ylim(0, top)
    for link, (t, s) in scores.items():
        if s.max() > top:
            k = s.argmax()
            s_ax.annotate(f"{link} peaks at {s.max():.0f}", (t[k], top), xytext=(t[k] + 1.5, top * 0.93),
                          fontsize=9, color=INK, va="top", arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
            top *= 0.9
    s_ax.set_ylim(0, 1.8 * np.percentile(allscores, 95))
    s_ax.grid(axis="y", color=GRID, lw=0.8)
    s_ax.legend(loc="upper right", frameon=False, ncol=n)
    for side in ("top", "right"):
        s_ax.spines[side].set_visible(False)

    for s0, s1, activity, zone in spans:
        if activity == "transition":
            s_ax.axvspan(s0, s1, color=GRID, alpha=0.7, lw=0)
        else:
            label = activity if zone == "none" else f"{activity}\nzone {zone}"
            ax[0].text((s0 + s1) / 2, 1.04, label, transform=ax[0].get_xaxis_transform(),
                       ha="center", va="bottom", fontsize=11)
        for axis in ax:
            axis.axvline(s0, color=SURFACE if axis is not s_ax else MUTED, lw=0.8)
    s_ax.text(0.0, -0.3, "Grey bands: 5 s transitions (cue given, not used for training).",
              transform=s_ax.transAxes, color=MUTED, fontsize=9)
    fig.suptitle(f"{name}: CSI amplitude and motion score per receiver", x=0.06, ha="left", fontsize=13, y=0.995)
    fig.subplots_adjust(left=0.08, right=0.9, top=0.9, bottom=0.11, hspace=0.12)
    out = ROOT / "plots" / f"{name}_session.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
