"""
csi_common.py - parsing and signal processing for ESP32 CSI captured with
Espressif's esp-csi `csi_recv_router` example (router = transmitter, ESP32 = receiver).

Line format printed by the firmware on ESP32 / ESP32-S3 / ESP32-C3 (24 fields before the data):
  CSI_DATA,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,
  fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,
  rx_format,len,first_word,"[v0,v1,...]"
On ESP32-C5 / C6 / C61 the firmware prints a shorter header (14 fields); both are handled.

Raw logs written by record.py contain one line per serial line:  <host_unix_time>\t<line>
"""
import numpy as np
from scipy.signal import butter, filtfilt, stft

# ----------------------------------------------------------------------------- parsing

# index of fields inside the header, by header length
_LAYOUTS = {
    24: dict(seq=1, mac=2, rssi=3, noise_floor=14, channel=16, ts=18),   # ESP32, S3, C3
    14: dict(seq=1, mac=2, rssi=3, noise_floor=5, channel=8, ts=9),      # C5, C6, C61
}


def parse_line(line):
    """Parse one CSI_DATA line. Returns a dict, or None if the line is not valid CSI."""
    if not line.startswith("CSI_DATA"):
        return None
    try:
        head, data = line.split(',"[', 1)
        data = data.strip().rstrip('"').rstrip(']')
        parts = head.split(',')
        lay = _LAYOUTS.get(len(parts))
        if lay is None:
            return None
        length = int(parts[-2])
        first_word_invalid = int(parts[-1])
        vals = np.array([int(v) for v in data.split(',')], dtype=np.int32)
        if len(vals) != length:          # truncated line (serial overflow) -> drop it
            return None
        return dict(
            seq=int(parts[lay['seq']]),
            mac=parts[lay['mac']],
            rssi=int(parts[lay['rssi']]),
            noise_floor=int(parts[lay['noise_floor']]),
            channel=int(parts[lay['channel']]),
            ts_us=int(parts[lay['ts']]),
            first_word_invalid=first_word_invalid,
            csi=vals,
        )
    except (ValueError, IndexError):
        return None


def load_raw_log(path):
    """Load a raw log written by record.py. Keeps only packets with the most common CSI length."""
    host_t, seq, rssi, ts, fwi, csi = [], [], [], [], [], []
    with open(path, 'r', errors='ignore') as f:
        for raw in f:
            if '\t' not in raw:
                continue
            t_str, line = raw.rstrip('\n').split('\t', 1)
            p = parse_line(line)
            if p is None:
                continue
            host_t.append(float(t_str)); seq.append(p['seq']); rssi.append(p['rssi'])
            ts.append(p['ts_us']); fwi.append(p['first_word_invalid']); csi.append(p['csi'])
    if not csi:
        raise ValueError(f"no CSI packets found in {path}")
    lengths = np.array([len(c) for c in csi])
    L = np.bincount(lengths).argmax()
    keep = lengths == L
    out = dict(
        host_t=np.array(host_t)[keep],
        seq=np.array(seq)[keep],
        rssi=np.array(rssi)[keep],
        ts_us=np.array(ts, dtype=np.int64)[keep],
        first_word_invalid=np.array(fwi)[keep],
        csi=np.stack([c for c, k in zip(csi, keep) if k]),
        dropped_other_length=int((~keep).sum()),
    )
    out['board_t'] = board_time_to_host(out['ts_us'], out['host_t'])
    return out


# ----------------------------------------------------------------------------- timing

def unwrap_ts(ts_us):
    """Board timestamps are 32-bit microseconds and wrap every ~71.6 minutes."""
    ts = ts_us.astype(np.int64)
    d = np.diff(ts)
    wraps = np.concatenate([[0], np.cumsum(d < -(1 << 31))])
    return ts + wraps * (1 << 32)


def board_time_to_host(ts_us, host_t):
    """Map each packet's board timestamp onto the host clock.

    Board timestamps are precise but have an unknown offset; host times are on a shared clock
    (so both boards line up) but jittery from USB buffering. A straight-line fit gives
    precise *and* shared times."""
    tb = unwrap_ts(ts_us) / 1e6
    if len(tb) < 10:
        return host_t.copy()
    a, b = np.polyfit(tb - tb[0], host_t, 1)
    return a * (tb - tb[0]) + b


# ----------------------------------------------------------------------------- CSI -> amplitude

def to_complex(csi, first_word_invalid=None):
    """int pairs -> complex, subcarriers in frequency order.

    Espressif stores each subcarrier as two signed bytes, imaginary part first, then real.
    For the 64 L-LTF subcarriers the buffer order is 0..31 then -32..-1; fftshift puts them
    in frequency order -32..31. (Check against Espressif's CSI docs if you change the CSI
    config, e.g. enable HT-LTF.)"""
    n, L = csi.shape
    v = csi.reshape(n, L // 2, 2).astype(np.float64)
    H = v[..., 1] + 1j * v[..., 0]
    if first_word_invalid is not None:
        H[first_word_invalid.astype(bool), :2] = np.nan
    if H.shape[1] == 64:
        H = np.fft.fftshift(H, axes=1)
    return H


def pick_subcarriers(amp, rel=0.15):
    """Indices of subcarriers that carry signal (drops DC and guard bands, which read ~0)."""
    m = np.nanmean(amp, axis=0)
    ref = np.nanmedian(m[m >= np.nanmedian(m)])
    return np.where(m > rel * ref)[0]


# ----------------------------------------------------------------------------- cleaning

def resample(t, X, t0, n, fs):
    """Linear interpolation of each column of X (packets x subcarriers) onto n samples from t0 at fs."""
    grid = t0 + np.arange(n) / fs
    order = np.argsort(t)
    t, X = t[order], X[order]
    t, uniq = np.unique(t, return_index=True)
    X = X[uniq]
    out = np.empty((n, X.shape[1]))
    for j in range(X.shape[1]):
        col = X[:, j]
        ok = np.isfinite(col)
        out[:, j] = np.interp(grid, t[ok], col[ok]) if ok.sum() >= 2 else np.nan
    return grid, out


def hampel(X, half_window=5, n_sigmas=3.0):
    """Replace spikes with the local median (per column). A spike is a sample more than
    n_sigmas robust standard deviations away from the median of its neighbourhood."""
    k = half_window
    Xp = np.pad(X, ((k, k), (0, 0)), mode='edge')
    win = np.lib.stride_tricks.sliding_window_view(Xp, 2 * k + 1, axis=0)   # (T, S, 2k+1)
    med = np.median(win, axis=-1)
    mad = 1.4826 * np.median(np.abs(win - med[..., None]), axis=-1)
    bad = np.abs(X - med) > n_sigmas * np.maximum(mad, 1e-9)
    Y = X.copy()
    Y[bad] = med[bad]
    return Y, bad


def gain_outliers(amp, half_window=5, thresh=0.3):
    """Find packets where *all* subcarriers jump together (a gain glitch, not motion).

    For each packet, compare every subcarrier with its own running median over neighbouring
    packets (in log scale) and take the median of those ratios across subcarriers. Motion moves
    subcarriers in different directions, so this stays near 0; a gain glitch shifts them all
    the same way. Returns a boolean mask of bad packets."""
    k = half_window
    L = np.log(np.where(np.isfinite(amp), amp, np.nan) + 1e-6)
    Lp = np.pad(L, ((k, k), (0, 0)), mode='edge')
    win = np.lib.stride_tricks.sliding_window_view(Lp, 2 * k + 1, axis=0)
    med = np.nanmedian(win, axis=-1)
    r = np.nanmedian(L - med, axis=1)
    spread = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r)))
    return np.abs(r) > max(thresh, 5 * spread)


def lowpass(X, fs, fc=20.0, order=4):
    """Zero-phase Butterworth low-pass. Human motion at 2.4 GHz sits well below ~20 Hz."""
    b, a = butter(order, fc / (fs / 2))
    return filtfilt(b, a, X, axis=0)


def process_clip(board, t0, t1, subcarriers, fs=100, n=None, fc=20.0):
    """Cut one clip from one board's log and clean it.
    Returns (amplitude array n x len(subcarriers), info dict)."""
    n = n or int(round((t1 - t0) * fs))
    t = board['board_t']
    sel = (t >= t0 - 0.5) & (t <= t1 + 0.5)            # small margin for interpolation
    inside = (t >= t0) & (t <= t1)
    if inside.sum() < 0.3 * n:
        return None, dict(n_packets=int(inside.sum()), ok=False)
    H = to_complex(board['csi'][sel], board['first_word_invalid'][sel])
    amp = np.abs(H)[:, subcarriers]
    glitch = gain_outliers(amp)                          # 1. drop whole-packet gain glitches
    _, A = resample(t[sel][~glitch], amp[~glitch], t0, n, fs)   # 2. exact 100 Hz grid
    A, spikes = hampel(A)                                # 3. single-subcarrier spikes
    A = lowpass(A, fs, fc)                               # 4. remove fast noise
    gaps = np.diff(t[inside]) if inside.sum() > 1 else np.array([t1 - t0])
    info = dict(n_packets=int(inside.sum()), expected=n, max_gap_s=float(gaps.max()),
                glitch_packets=int(glitch.sum()), spike_frac=float(spikes.mean()),
                ok=bool(inside.sum() >= 0.8 * n and gaps.max() < 0.25))
    return A, info


# ----------------------------------------------------------------------------- features for plots

def motion_component(A):
    """First principal component of the amplitude changes: one time series that follows the
    strongest motion across all subcarriers."""
    Z = A - A.mean(axis=0)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    return Z @ Vt[0]


def spectrogram(x, fs, nperseg=64, noverlap=56):
    f, t, S = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, np.abs(S)
