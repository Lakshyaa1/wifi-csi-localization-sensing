"""Static vs motion view of a logger.py recording (wave and walk merged into "motion").

Usage:
  python static_vs_motion.py first_motion
Works on 0.5 s windows instead of single packets: each window becomes one data
point described by how much each subcarrier fluctuated inside it.
Produces plots/<name>_static_vs_motion.png.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_csi import load, phase_spans

STILL, MOTION = "#4c78a8", "#e45756"


def main():
    name = sys.argv[1]
    t, labels, amps, _ = load(name)
    labels = np.array(labels)
    a = amps[:, amps.std(axis=0) > 0]
    is_motion = labels != "still"

    # Non-overlapping 0.5 s windows -> one data point each
    win = 0.5
    edges = np.arange(t[0], t[-1], win)
    feats, wlab, wt = [], [], []
    for s0 in edges:
        m = (t >= s0) & (t < s0 + win)
        if m.sum() < 10:
            continue
        feats.append(a[m].std(axis=0))           # per-subcarrier fluctuation in the window
        wlab.append(is_motion[m].mean() > 0.5)
        wt.append(s0 + win / 2)
    feats, wlab, wt = np.array(feats), np.array(wlab), np.array(wt)
    score = feats.mean(axis=1)

    # Simple threshold detector: midpoint between class medians (log scale)
    thr = np.exp((np.log(np.median(score[~wlab])) + np.log(np.median(score[wlab]))) / 2)
    pred = score > thr
    acc = (pred == wlab).mean()
    tpr, tnr = pred[wlab].mean(), (~pred[~wlab]).mean()

    # PCA on log window features
    z = np.log(feats)
    z = (z - z.mean(0)) / z.std(0)
    _, sv, vt = np.linalg.svd(z, full_matrices=False)
    pcs = z @ vt[:2].T
    var = sv**2 / (sv**2).sum()

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 1.4])
    spans = phase_spans(t, list(np.where(is_motion, "motion", "still")))

    ax0 = fig.add_subplot(gs[0, :])
    im = ax0.imshow(a.T, aspect="auto", origin="lower", cmap="viridis", extent=[t[0], t[-1], 0, a.shape[1]])
    for s0, s1, lab in spans:
        ax0.axvline(s0, color="w", lw=1, ls="--")
        ax0.text((s0 + s1) / 2, a.shape[1] + 1, lab, ha="center", color=MOTION if lab == "motion" else STILL,
                 fontsize=11, fontweight="bold")
    ax0.set_ylabel("subcarrier")
    ax0.set_title("CSI amplitude: 52 subcarriers over time", pad=18)
    fig.colorbar(im, cax=ax0.inset_axes([1.005, 0, 0.01, 1]), label="|H|")

    ax1 = fig.add_subplot(gs[1, :], sharex=ax0)
    ax1.scatter(wt, score, c=np.where(wlab, MOTION, STILL), s=18, zorder=3)
    ax1.plot(wt, score, color="0.6", lw=0.8)
    ax1.axhline(thr, color="k", ls="--", lw=1, label=f"threshold {thr:.2f}")
    for s0, s1, lab in spans:
        ax1.axvspan(s0, s1, color=MOTION if lab == "motion" else STILL, alpha=0.08, lw=0)
    ax1.set_yscale("log")
    ax1.set_ylabel("motion score\n(mean std per 0.5 s)")
    ax1.set_xlabel("time (s)")
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(gs[2, 0])
    bins = np.logspace(np.log10(score.min() * 0.9), np.log10(score.max() * 1.1), 40)
    ax2.hist(score[~wlab], bins=bins, color=STILL, alpha=0.75, label=f"still ({(~wlab).sum()} windows)")
    ax2.hist(score[wlab], bins=bins, color=MOTION, alpha=0.75, label=f"motion ({wlab.sum()} windows)")
    ax2.axvline(thr, color="k", ls="--", lw=1)
    ax2.set_xscale("log")
    ax2.set_xlabel("motion score (log)")
    ax2.set_ylabel("number of windows")
    ax2.set_title(f"Threshold detector: {acc:.0%} correct (motion caught {tpr:.0%}, still kept {tnr:.0%})")
    ax2.legend()

    ax3 = fig.add_subplot(gs[2, 1])
    ax3.scatter(pcs[~wlab, 0], pcs[~wlab, 1], s=22, color=STILL, alpha=0.7, label="still")
    ax3.scatter(pcs[wlab, 0], pcs[wlab, 1], s=22, color=MOTION, alpha=0.7, label="motion")
    ax3.set_xlabel(f"PC1 ({var[0]:.0%} of variance)")
    ax3.set_ylabel(f"PC2 ({var[1]:.0%})")
    ax3.set_title("Each 0.5 s window as a data point (PCA)")
    ax3.legend()

    fig.tight_layout()
    out = Path(__file__).parent / "plots" / f"{name}_static_vs_motion.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    print(f"windows: {len(wlab)} ({(~wlab).sum()} still, {wlab.sum()} motion)")
    print(f"median score still {np.median(score[~wlab]):.2f}, motion {np.median(score[wlab]):.2f} "
          f"(x{np.median(score[wlab]) / np.median(score[~wlab]):.1f})")
    print(f"threshold accuracy {acc:.1%}, motion detected {tpr:.1%}, still correct {tnr:.1%}")
    missed = wt[wlab & ~pred]
    print("missed motion windows at t =", np.round(missed, 1).tolist())


if __name__ == "__main__":
    main()
