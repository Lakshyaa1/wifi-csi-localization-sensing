"""Raw vs sanitized CSI phase for a logger.py recording.

Usage:
  python phase_check.py first_motion
ESP32 LLTF subcarriers come in the order 0..31, -32..-1; we reorder them to
-32..31 so phase can be unwrapped across frequency. Sanitization removes the
per-packet linear phase trend across subcarriers (slope = timing offset,
intercept = random phase offset), the standard fix for single-antenna CSI.
Produces plots/<name>_phase.png.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_csi import phase_spans


def load_complex(name):
    path = Path(__file__).parent / "data" / f"{name}.csv"
    t, labels, H = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            v = np.array(row["csi"].split(), dtype=float)[:128]
            H.append(v[1::2] + 1j * v[0::2])  # imaginary first in ESP32 output
            t.append(float(row["host_time"]))
            labels.append(row["label"])
    H = np.array(H)
    order = np.r_[32:64, 0:32]  # -> subcarrier index -32..31
    H = H[:, order]
    k = np.arange(-32, 32)
    active = np.abs(H).std(axis=0) > 0
    return np.array(t) - t[0], np.array(labels), H[:, active], k[active]


def sanitize(H, k):
    ph = np.unwrap(np.angle(H), axis=1)
    out = np.empty_like(ph)
    for i in range(len(ph)):
        slope, icept = np.polyfit(k, ph[i], 1)
        out[i] = ph[i] - (slope * k + icept)
    return ph, out


def main():
    name = sys.argv[1]
    t, labels, H, k = load_complex(name)
    raw, clean = sanitize(H, k)
    amp = np.abs(H)
    sc = len(k) // 4  # one representative subcarrier
    spans = phase_spans(t, list(np.where(labels == "still", "still", "motion")))

    def shade(ax):
        for s0, s1, lab in spans:
            ax.axvspan(s0, s1, color="#e45756" if lab == "motion" else "#4c78a8", alpha=0.08, lw=0)

    fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    ax[0].plot(t, amp[:, sc], lw=0.6, color="tab:green")
    ax[0].set_ylabel("|H|")
    ax[0].set_title(f"Subcarrier {k[sc]}: amplitude")
    ax[1].plot(t, np.angle(H[:, sc]), ".", ms=1.5, color="0.3")
    ax[1].set_ylabel("raw phase (rad)")
    ax[1].set_title("Raw phase: random every packet (clock offsets), useless as-is")
    ax[2].plot(t, clean[:, sc], lw=0.6, color="tab:purple")
    ax[2].set_ylabel("sanitized phase (rad)")
    ax[2].set_title("Sanitized phase (linear trend across subcarriers removed per packet)")
    im = ax[3].imshow(clean.T, aspect="auto", origin="lower", cmap="twilight",
                      extent=[t[0], t[-1], k[0], k[-1]], vmin=-np.pi / 2, vmax=np.pi / 2)
    ax[3].set_ylabel("subcarrier index")
    ax[3].set_xlabel("time (s)")
    ax[3].set_title("Sanitized phase, all subcarriers")
    cax = ax[3].inset_axes([1.005, 0, 0.01, 1])  # keep all panels the same width
    fig.colorbar(im, cax=cax, label="rad")
    for a in ax[:3]:
        shade(a)
    fig.tight_layout()
    out = Path(__file__).parent / "plots" / f"{name}_phase.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")

    still = labels == "still"
    # circular spread of raw phase vs std of sanitized phase, still vs motion
    def circ_std(x):
        r = np.abs(np.mean(np.exp(1j * x)))
        return np.sqrt(-2 * np.log(max(r, 1e-12)))
    print(f"raw phase circular std  (subcarrier {k[sc]}): still {circ_std(np.angle(H[still, sc])):.2f} rad, "
          f"motion {circ_std(np.angle(H[~still, sc])):.2f} rad   (uniform random would be > 2)")
    print(f"sanitized phase std, mean over subcarriers: still {clean[still].std(axis=0).mean():.3f} rad, "
          f"motion {clean[~still].std(axis=0).mean():.3f} rad")
    print(f"amplitude std, mean over subcarriers:        still {amp[still].std(axis=0).mean():.3f}, "
          f"motion {amp[~still].std(axis=0).mean():.3f}")


if __name__ == "__main__":
    main()
