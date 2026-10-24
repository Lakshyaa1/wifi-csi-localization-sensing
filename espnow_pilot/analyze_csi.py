"""Deeper look at a logger.py recording: each packet as a data point.

Usage:
  python analyze_csi.py first_motion
Produces plots/<name>_analysis.png with:
  1. PCA scatter: every packet's 52-subcarrier amplitude vector projected to 2D,
     coloured by label (do still / wave / walk form separate clusters?)
  2. RSSI vs CSI motion score over time (how much more CSI sees than RSSI)
  3. Doppler-style spectrogram of the dominant CSI component (how fast it ripples)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_csi import load, phase_spans

COLORS = {"still": "#4c78a8", "wave": "#e45756", "walk": "#54a24b", "test": "#999999"}


def main():
    name = sys.argv[1]
    t, labels, amps, rssi = load(name)
    labels = np.array(labels)
    a = amps[:, amps.std(axis=0) > 0]

    # 1. PCA of per-packet amplitude vectors (centred, scaled per subcarrier)
    z = (a - a.mean(axis=0)) / a.std(axis=0)
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    pcs = z @ vt[:2].T
    var = s**2 / (s**2).sum()

    # 2. Motion score from CSI vs the same statistic on RSSI
    rate = len(t) / (t[-1] - t[0])
    win = max(3, int(round(0.5 * rate)))
    csi_score = np.array([a[max(0, i - win + 1): i + 1].std(axis=0).mean() for i in range(len(a))])
    rssi_score = np.array([rssi[max(0, i - win + 1): i + 1].std() for i in range(len(rssi))])

    # 3. Spectrogram of PC1 resampled onto a uniform grid
    fs = 70.0
    tu = np.arange(t[0], t[-1], 1 / fs)
    pc1 = np.interp(tu, t, pcs[:, 0])
    nper, hop = int(fs), int(fs / 10)
    win_fn = np.hanning(nper)
    frames = [pc1[i:i + nper] - pc1[i:i + nper].mean() for i in range(0, len(pc1) - nper, hop)]
    spec = np.abs(np.fft.rfft(np.array(frames) * win_fn, axis=1)) ** 2
    freqs = np.fft.rfftfreq(nper, 1 / fs)
    tf = tu[np.arange(len(frames)) * hop + nper // 2]

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.4], height_ratios=[1, 1, 1])

    ax = fig.add_subplot(gs[:, 0])
    for lab in dict.fromkeys(labels):
        m = labels == lab
        ax.scatter(pcs[m, 0], pcs[m, 1], s=4, alpha=0.45, color=COLORS.get(lab), label=f"{lab} (n={m.sum()})")
    ax.set_xlabel(f"PC1 ({var[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.0%} of variance)")
    ax.set_title("Every packet as a data point (PCA of 52-subcarrier amplitude)")
    ax.legend(markerscale=4)

    spans = phase_spans(t, list(labels))
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t, rssi, lw=0.6, color="0.3")
    ax1.set_ylabel("RSSI (dBm)")
    ax1.set_title("RSSI per packet")
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax2.plot(t, csi_score / np.median(csi_score[labels == "still"]), lw=1, color="k", label="CSI motion score")
    ax2.plot(t, rssi_score / np.median(rssi_score[labels == "still"]), lw=1, color="tab:orange", alpha=0.8,
             label="RSSI motion score")
    ax2.axhline(1, ls=":", color="0.5")
    ax2.set_ylabel("score / still median")
    ax2.legend(loc="upper right", fontsize=8)
    for axis in (ax1, ax2):
        for s0, s1, lab in spans:
            axis.axvspan(s0, s1, color=COLORS.get(lab), alpha=0.12, lw=0)

    ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
    ax3.pcolormesh(tf, freqs, 10 * np.log10(spec.T + 1e-9), shading="auto", cmap="magma")
    ax3.set_ylim(0, 20)
    ax3.set_ylabel("frequency (Hz)")
    ax3.set_xlabel("time (s)")
    ax3.set_title("How fast the CSI ripples (spectrogram of PC1)")
    for s0, s1, lab in spans:
        ax3.text((s0 + s1) / 2, 18, lab, ha="center", color="w", fontsize=9)

    fig.tight_layout()
    out = Path(__file__).parent / "plots" / f"{name}_analysis.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    print(f"PCA: PC1 {var[0]:.1%}, PC2 {var[1]:.1%} of variance")
    for lab in dict.fromkeys(labels):
        m = labels == lab
        print(f"  {lab:>6}: PC1 mean {pcs[m,0].mean():+.2f} sd {pcs[m,0].std():.2f} | "
              f"CSI score x{np.median(csi_score[m]) / np.median(csi_score[labels == 'still']):.2f}, "
              f"RSSI score x{np.median(rssi_score[m]) / np.median(rssi_score[labels == 'still']):.2f} (vs still)")
    band = (freqs >= 1) & (freqs <= 10)
    for lab in ("still", "wave", "walk"):
        idx = [i for i, tt in enumerate(tf) if labels[np.searchsorted(t, tt).clip(0, len(t) - 1)] == lab]
        if idx:
            p = spec[idx][:, band].mean(axis=0)
            print(f"  {lab:>6}: peak ripple frequency {freqs[band][p.argmax()]:.1f} Hz, 1-10 Hz power {10*np.log10(p.sum()):.1f} dB")


if __name__ == "__main__":
    main()
