"""Turn a logger.py recording into fixed-size, labelled windows for models.

Usage:
  python preprocess.py s01_roomA [s02_roomA ...]
Writes data/processed/<name>.npz with
  X        float32 [windows, links*2, time, subcarriers]
           channels per link: amplitude, sanitized phase (link order = `links`)
  activity str     [windows]
  zone     str     [windows]   "1".."6", or "none"
  t0       float   [windows]   window start (s from recording start)
  session, room    str         recording name; room = part after the first "_"
  links    str     [links]
  subcarriers int  [subcarriers]  LLTF subcarrier indices (-32..31, nulls dropped)

Steps (per recording):
  1. join the links on the transmitter's packet id and keep packets every receiver
     heard (same id = same instant); drop "transition" packets; time axis from the
     first link's device timestamp (us)
  2. Hampel filter on amplitude and phase (replace spikes > 3 MAD within +-3 samples)
  3. resample every labelled phase onto a uniform FS grid (linear interpolation)
  4. 4th-order Butterworth low-pass at CUTOFF Hz (zero-phase), amplitude and phase
  5. per-channel, per-subcarrier z-score using this recording's statistics
     (unlabelled, so it is also allowed on a target room later), clipped to +-CLIP
  6. WIN-second windows with 50% overlap, only inside one phase (never mixing labels)
Only the 64-subcarrier legacy LTF is used for now; the HT-LTF part is kept in the
raw CSV for later.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from phase_check import sanitize

FS = 70.0        # Hz, uniform resampling rate (~ measured packet rate)
CUTOFF = 20.0    # Hz, low-pass
WIN = 2.0        # s
HOP = 1.0        # s (50% overlap)
CLIP = 5.0       # z-score clip after normalisation
DATA = Path(__file__).parent / "data"


def read_recording(name):
    """Joined complex LLTF CSI [packets, links, subcarriers] + labels + device time.

    Subcarrier order is -32..31 with nulls dropped. Recordings from the first
    logger version (one link, `label` column, no packet id) are read as link rx1.
    """
    per_link = {}
    with open(DATA / f"{name}.csv") as f:
        for i, r in enumerate(csv.DictReader(f)):
            v = np.array(r["csi"].split(), dtype=float)[:128]
            h = (v[1::2] + 1j * v[0::2])[np.r_[32:64, 0:32]]   # imaginary first; -> -32..31
            per_link.setdefault(r.get("link", "rx1"), {})[int(r.get("id", i))] = (
                int(r["local_timestamp"]), r.get("activity", r.get("label")), r.get("zone", "none"), h)
    links = sorted(per_link)
    ids = sorted(set.intersection(*(set(d) for d in per_link.values())))
    heard = max(len(d) for d in per_link.values())
    first = per_link[links[0]]
    ts = np.array([first[i][0] for i in ids], dtype=float)
    d = np.diff(ts)
    d[d < 0] += 2**32          # 32-bit microsecond counter wrap
    t = np.r_[0, np.cumsum(d)] / 1e6
    activity = np.array([first[i][1] for i in ids])
    zone = np.array([first[i][2] for i in ids])
    H = np.array([[per_link[l][i][3] for l in links] for i in ids])   # [packets, links, 64]
    k = np.arange(-32, 32)
    active = (np.abs(H).std(axis=0) > 0).all(axis=0)
    keep = activity != "transition"
    return t[keep], activity[keep], zone[keep], H[keep][:, :, active], k[active], links, 1 - len(ids) / heard


def hampel(x, half=3, n_sigma=3.0):
    """Replace outliers along axis 0 with the local median (per column)."""
    x = x.copy()
    n = len(x)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        med = np.median(x[lo:hi], axis=0)
        mad = 1.4826 * np.median(np.abs(x[lo:hi] - med), axis=0)
        bad = np.abs(x[i] - med) > n_sigma * np.maximum(mad, 1e-9)
        x[i, bad] = med[bad]
    return x


def segments(activity, zone):
    """(start, end) index ranges of consecutive packets with the same activity and zone."""
    key = np.char.add(np.char.add(activity, "/"), zone)
    cuts = np.r_[0, np.where(key[1:] != key[:-1])[0] + 1, len(key)]
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:])]


def process(name):
    t, activity, zone, H, k, links, loss = read_recording(name)
    amp = hampel(np.abs(H))
    phase = np.stack([sanitize(H[:, l], k)[1] for l in range(len(links))], axis=1)
    phase = hampel(phase)  # phase is undefined during deep fades -> spikes
    # [packets, links*2, subcarriers]: amplitude and phase of link 0, then link 1, ...
    chans = np.stack([amp, phase], axis=2).reshape(len(t), 2 * len(links), len(k))
    b, a = butter(4, CUTOFF / (FS / 2))

    # resample + filter per phase, collect the continuous signals
    pieces = []
    for s0, s1 in segments(activity, zone):
        if t[s1 - 1] - t[s0] < WIN:
            continue
        grid = np.arange(t[s0], t[s1 - 1], 1 / FS)
        sig = np.stack([np.stack([np.interp(grid, t[s0:s1], chans[s0:s1, c, j]) for j in range(len(k))], 1)
                        for c in range(chans.shape[1])])             # [channels, time, sc]
        if sig.shape[1] > 3 * 4 * 2:                                 # filtfilt needs > padlen
            sig = filtfilt(b, a, sig, axis=1)
        pieces.append((grid, sig, activity[s0], zone[s0]))

    # per-channel, per-subcarrier z-score with this recording's statistics
    allsig = np.concatenate([p[1] for p in pieces], axis=1)
    mu, sd = allsig.mean(axis=1, keepdims=True), allsig.std(axis=1, keepdims=True) + 1e-6

    n, h = int(round(WIN * FS)), int(round(HOP * FS))
    X, act, zon, t0 = [], [], [], []
    for grid, sig, a_, z_ in pieces:
        sig = np.clip((sig - mu) / sd, -CLIP, CLIP)  # deep fades -> bounded
        for i in range(0, sig.shape[1] - n + 1, h):
            X.append(sig[:, i:i + n])
            act.append(a_)
            zon.append(z_)
            t0.append(grid[i])
    X = np.array(X, dtype=np.float32)
    out = DATA / "processed" / f"{name}.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, X=X, activity=np.array(act), zone=np.array(zon), t0=np.array(t0),
                        session=name, room=name.split("_", 1)[1] if "_" in name else "",
                        links=np.array(links), subcarriers=k)
    return out, X, np.array(act), np.array(zon), t, links, loss


def main():
    for name in sys.argv[1:]:
        out, X, act, zon, t, links, loss = process(name)
        print(f"{name}: links {links}, {len(t)} joined packets over {t[-1]:.1f} s "
              f"({100 * loss:.1f}% lost in the join) -> {X.shape[0]} windows of shape {X.shape[1:]} "
              f"(channels, time, subcarriers) -> {out}")
        print(f"  NaNs: {int(np.isnan(X).sum())}")
        for z in dict.fromkeys(zon):
            m = zon == z
            print(f"  zone {z}: " + ", ".join(f"{a} {int((act[m] == a).sum())}" for a in dict.fromkeys(act[m])))


if __name__ == "__main__":
    main()
