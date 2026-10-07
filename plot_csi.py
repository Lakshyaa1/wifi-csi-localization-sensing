"""Plot a CSI recording made by logger.py.

Usage:
  python plot_csi.py wave_test
Produces plots/<name>.png with:
  1. amplitude heatmap (time x subcarrier) from the legacy LTF (64 subcarriers)
  2. amplitude of a few subcarriers over time
  3. motion score: per-subcarrier std over a 0.5 s sliding window, averaged
Phase boundaries from the labels are shaded.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LLTF_SUBCARRIERS = 64


def load(name):
    path = Path(__file__).parent / "data" / f"{name}.csv"
    t, labels, amps, rssi = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            vals = np.array(row["csi"].split(), dtype=float)
            if len(vals) < 2 * LLTF_SUBCARRIERS:
                continue
            lltf = vals[: 2 * LLTF_SUBCARRIERS]
            imag, real = lltf[0::2], lltf[1::2]  # ESP32 order: imaginary first
            amps.append(np.hypot(real, imag))
            t.append(float(row["host_time"]))
            labels.append(row["label"])
            rssi.append(int(row["rssi"]))
    t = np.array(t) - t[0]
    return t, labels, np.array(amps), np.array(rssi)


def phase_spans(t, labels):
    spans, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            spans.append((t[start], t[i - 1], labels[start]))
            start = i
    return spans


def main():
    name = sys.argv[1]
    t, labels, amps, rssi = load(name)
    active = amps.std(axis=0) > 0  # drop null / guard subcarriers
    a = amps[:, active]
    sc_idx = np.where(active)[0]

    rate = len(t) / (t[-1] - t[0])
    win = max(3, int(round(0.5 * rate)))
    score = np.array([a[max(0, i - win + 1): i + 1].std(axis=0).mean() for i in range(len(a))])

    spans = phase_spans(t, labels)
    colors = {lab: c for lab, c in zip(sorted(set(labels)), ["#d9e8f5", "#f8d9d9", "#dff2df", "#f5ecd9"])}

    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1.2, 1]})
    im = ax[0].imshow(a.T, aspect="auto", origin="lower", cmap="viridis",
                      extent=[t[0], t[-1], 0, a.shape[1]])
    ax[0].set_ylabel("active subcarrier #")
    ax[0].set_title(f"{name}: CSI amplitude (LLTF), {rate:.0f} packets/s")
    fig.colorbar(im, cax=ax[0].inset_axes([1.005, 0, 0.01, 1]), label="|H|")

    for k in np.linspace(0, a.shape[1] - 1, 4).astype(int):
        ax[1].plot(t, a[:, k], lw=0.8, label=f"subcarrier {sc_idx[k]}")
    ax[1].set_ylabel("|H|")
    ax[1].legend(loc="upper right", fontsize=8, ncol=4)

    ax[2].plot(t, score, color="k", lw=1)
    ax[2].set_ylabel("motion score\n(0.5 s std)")
    ax[2].set_xlabel("time (s)")

    for s0, s1, lab in spans:
        for axis in ax[1:]:
            axis.axvspan(s0, s1, color=colors[lab], alpha=0.6, lw=0)
        ax[2].text((s0 + s1) / 2, ax[2].get_ylim()[1] * 0.9, lab, ha="center", fontsize=10)

    out = Path(__file__).parent / "plots" / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    print(f"packets: {len(t)}, rate {rate:.1f}/s, RSSI mean {rssi.mean():.1f} dBm")
    for lab in dict.fromkeys(labels):
        m = np.array(labels) == lab
        print(f"  {lab:>8}: motion score mean {score[m].mean():.2f}, median {np.median(score[m]):.2f}")


if __name__ == "__main__":
    main()
