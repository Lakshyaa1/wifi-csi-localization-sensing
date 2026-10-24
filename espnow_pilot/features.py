"""Handcrafted window features for the classical baselines."""
import numpy as np

from preprocess import FS

BANDS = [(0.3, 2), (2, 5), (5, 10), (10, 20)]  # Hz


def window_features(X):
    """X [windows, links*2, time, subcarriers] -> [windows, features].

    Per subcarrier and channel: mean (where the pattern sits: position), std (how
    much it moves: activity) and mean absolute step (how fast it moves). Per link,
    from the amplitude averaged over subcarriers: energy in four frequency bands,
    dominant frequency, and end-minus-start level.
    """
    n = len(X)
    mean = X.mean(axis=2)
    std = X.std(axis=2)
    speed = np.abs(np.diff(X, axis=2)).mean(axis=2)
    common = X[:, 0::2].mean(axis=3)                                  # [windows, links, time]
    spec = np.abs(np.fft.rfft(common - common.mean(axis=2, keepdims=True), axis=2)) ** 2
    f = np.fft.rfftfreq(X.shape[2], 1 / FS)
    energy = np.stack([spec[:, :, (f >= lo) & (f < hi)].sum(axis=2) for lo, hi in BANDS], axis=2)
    dom = f[spec.argmax(axis=2)]
    q = X.shape[2] // 4
    slope = common[:, :, -q:].mean(axis=2) - common[:, :, :q].mean(axis=2)
    return np.concatenate([mean.reshape(n, -1), std.reshape(n, -1), speed.reshape(n, -1),
                           np.log1p(energy).reshape(n, -1), dom, slope], axis=1)
