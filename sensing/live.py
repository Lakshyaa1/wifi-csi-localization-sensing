"""
live.py - live activity guess from one or two boards, shown full screen.

    python live.py dataset.npz --ports /dev/serial/by-id/...A /dev/serial/by-id/...B

Learns the simple baseline (train_baseline.features + logistic regression) from every clip in
dataset.npz, then every half second takes the last 5 s from each board, cleans it exactly like
build_dataset.py does, and shows the most likely class with its probability.
Give the ports in the same order as when the dataset was recorded (board0 first).
Esc or q closes the window.
"""
import argparse, collections, threading, time
import numpy as np
import tkinter as tk
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from csi_common import parse_line, board_time_to_host, process_clip
from build_dataset import DATA_SUBCARRIERS
from train_baseline import features

COLORS = {"empty": "#3c4043", "sit": "#b06000", "walk": "#1e8e3e", "wave": "#1967d2"}


class Board(threading.Thread):
    """Reads one serial port and keeps the last few seconds of parsed packets."""
    def __init__(self, port, baud, keep_sec=8.0):
        super().__init__(daemon=True)
        import serial
        self.ser = serial.Serial()
        self.ser.port, self.ser.baudrate, self.ser.timeout = port, baud, 0.2
        self.ser.dtr = self.ser.rts = False   # opening with these set resets the board
        self.ser.open()
        self.buf = collections.deque()
        self.keep_sec = keep_sec
        self.first_t = None     # host time of the first packet after the board (re)started
        self.lock = threading.Lock()

    def run(self):
        while True:
            raw = self.ser.readline()
            if not raw:
                continue
            p = parse_line(raw.decode("ascii", errors="ignore").strip())
            if p is None or len(p["csi"]) != 128:
                continue
            t = time.time()
            with self.lock:
                if self.first_t is None:
                    self.first_t = t
                self.buf.append((t, p["ts_us"], p["first_word_invalid"], p["csi"]))
                while self.buf and self.buf[0][0] < t - self.keep_sec:
                    self.buf.popleft()

    def snapshot(self):
        """Buffer in the same form as csi_common.load_raw_log returns."""
        with self.lock:
            rows = list(self.buf)
        if len(rows) < 10:
            return None
        host_t = np.array([r[0] for r in rows])
        ts_us = np.array([r[1] for r in rows], dtype=np.int64)
        return dict(host_t=host_t, ts_us=ts_us,
                    first_word_invalid=np.array([r[2] for r in rows]),
                    csi=np.stack([r[3] for r in rows]),
                    board_t=board_time_to_host(ts_us, host_t))


def window_features(boards, t1, clip_sec, fs, settle_sec=5.0):
    """Features for the clip_sec ending at t1, or (None, reason)."""
    t0 = t1 - clip_sec
    n = int(clip_sec * fs)
    arrs = []
    for b in boards:
        snap = b.snapshot() if hasattr(b, "snapshot") else b
        if snap is None:
            return None, "waiting for packets"
        # the firmware locks the receiver gain after its first 100 packets; that one-off step
        # looks like "sit", so skip the first seconds after a board starts
        first = getattr(b, "first_t", None)
        if first is not None and t0 < first + settle_sec:
            return None, f"settling ({first + settle_sec - t0:.0f} s)"
        A, info = process_clip(snap, t0, t1, DATA_SUBCARRIERS, fs=fs, n=n)
        if A is None or info["n_packets"] < 0.8 * n:
            return None, f"too few packets ({info['n_packets']} of {n})"
        arrs.append(A.T)
    return features(np.stack(arrs), fs), "ok"


def train(dataset):
    d = np.load(dataset)
    X, y, fs = d["X"], d["y"], int(d["fs"])
    labels = [str(l) for l in d["labels"]]
    F = np.stack([features(c, fs) for c in X])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(F, y)
    return clf, labels, fs, X.shape[-1] / fs, X.shape[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--ports", nargs="+", required=True)
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--every", type=float, default=0.5, help="seconds between guesses")
    a = ap.parse_args()

    clf, labels, fs, clip_sec, n_boards = train(a.dataset)
    if len(a.ports) != n_boards:
        raise SystemExit(f"dataset has {n_boards} boards but {len(a.ports)} ports were given")
    print(f"learned from {a.dataset}: classes {labels}, {clip_sec:g} s windows")
    boards = [Board(p, a.baud) for p in a.ports]
    for b in boards:
        b.start()

    state = {"label": "starting", "probs": None, "note": "filling the first 5 s"}

    def guess_loop():
        while True:
            time.sleep(a.every)
            F, why = window_features(boards, time.time() - 0.3, clip_sec, fs)   # 0.3 s margin for USB lag
            if F is None:
                state.update(label="...", probs=None, note=why)
                continue
            p = clf.predict_proba(F[None])[0]
            state.update(label=labels[int(p.argmax())], probs=p, note="")
            print(time.strftime("%H:%M:%S"), "  ".join(f"{l} {q:.2f}" for l, q in zip(labels, p)))

    threading.Thread(target=guess_loop, daemon=True).start()

    root = tk.Tk()
    root.title("CSI live")
    root.attributes("-fullscreen", True)
    root.bind("<Escape>", lambda e: root.destroy())
    root.bind("q", lambda e: root.destroy())
    big = tk.Label(root, font=("Helvetica", 180, "bold"), fg="white")
    bars = tk.Label(root, font=("Courier", 40), fg="white", justify="left")
    note = tk.Label(root, font=("Helvetica", 28), fg="white")
    for w in (big, bars, note):
        w.pack(expand=True)

    def refresh():
        bg = COLORS.get(state["label"], "#202124")
        for w in (root, big, bars, note):
            w.configure(bg=bg)
        big.configure(text=state["label"].upper())
        if state["probs"] is not None:
            bars.configure(text="\n".join(f"{l:6} {'#' * int(round(q * 20)):20} {q * 100:3.0f}%"
                                          for l, q in zip(labels, state["probs"])))
        else:
            bars.configure(text="")
        note.configure(text=state["note"])
        root.after(100, refresh)

    refresh()
    root.mainloop()


if __name__ == "__main__":
    main()
