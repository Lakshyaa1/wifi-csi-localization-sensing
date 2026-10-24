"""Live view of CSI from every connected receiver: watch what your movement does.

Usage:
  python live_view.py                 # receivers from boards.csv that are plugged in
  python live_view.py --links rx1=/dev/ttyACM0
Stay away from the boards for the first 5 s: that is taken as the empty-room level.

The window shows, updating a few times per second:
  - top: activity level per receiver over the last 30 s, as a multiple of the
    empty-room level (1x = nothing moving)
  - below: one amplitude picture per receiver for the last 10 s (time left to
    right, subcarriers bottom to top, darker = stronger than usual)
Nothing is saved. Close the window to stop.
"""
import argparse
import sys
import threading
import time
from collections import deque

import matplotlib
import numpy as np
import serial

from logger import find_links, parse_line

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]     # one per receiver, fixed order
CALIBRATE = 5.0    # s of empty room at the start
HISTORY = 30.0     # s of activity level shown
PICTURE = 10.0     # s of amplitude shown
WINDOW = 0.5       # s used for one activity value


class Link:
    """Reads one receiver in the background and keeps its recent amplitudes."""

    def __init__(self, name, port, baud):
        self.name = name
        self.packets = deque(maxlen=4000)      # (time, amplitude of the 64 LLTF subcarriers)
        self.level = deque(maxlen=4000)        # (time, activity value)
        self.baseline = None
        self.s = serial.Serial()
        self.s.port, self.s.baudrate, self.s.timeout = port, baud, 0.5
        self.s.dtr = False
        self.s.rts = False
        self.s.open()
        threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        while True:
            parsed = parse_line(self.s.readline().decode(errors="ignore").strip())
            if parsed is None:
                continue
            v = np.array(parsed[1][:128], dtype=float)
            self.packets.append((time.time(), np.hypot(v[0::2], v[1::2])))

    def recent(self, seconds, now):
        rows = [p for p in list(self.packets) if now - p[0] <= seconds]
        if len(rows) < 5:
            return None, None
        a = np.array([r[1] for r in rows])
        return np.array([r[0] for r in rows]) - now, a[:, a.std(axis=0) > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", help="name=port pairs (default: receivers from boards.csv that are plugged in)")
    ap.add_argument("--baud", type=int, default=2000000)
    ap.add_argument("--backend", default="TkAgg")
    args = ap.parse_args()
    matplotlib.use(args.backend)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import LinearSegmentedColormap

    ports = find_links(args.links)
    if not ports:
        sys.exit("no receivers found: add them to boards.csv or pass --links")
    links = [Link(name, port, args.baud) for name, port in ports.items()]
    blues = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "toolbar": "None"})
    n = len(links)
    fig = plt.figure(figsize=(11, 3.4 + 2.3 * n), facecolor=SURFACE)
    fig.canvas.manager.set_window_title("Live CSI")
    grid = fig.add_gridspec(n + 1, 1, height_ratios=[1.5] + [1] * n, hspace=0.45,
                            left=0.09, right=0.97, top=0.9, bottom=0.08)
    top = fig.add_subplot(grid[0])
    top.set_facecolor(SURFACE)
    top.set_xlim(-HISTORY, 0)
    top.set_ylim(0, 4)
    top.set_ylabel("activity level\n(× empty room)")
    top.axhline(1, color=MUTED, lw=1, ls=(0, (4, 3)))
    top.text(-HISTORY + 0.3, 1.08, "empty-room level", color=MUTED, fontsize=9)
    top.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        top.spines[side].set_visible(False)
    lines = [top.plot([], [], color=c, lw=2, label=l.name)[0] for l, c in zip(links, SERIES)]
    top.legend(loc="upper left", frameon=False, ncol=n)
    status = fig.text(0.09, 0.945, "", fontsize=14, weight="bold")

    images = []
    for i, link in enumerate(links):
        ax = fig.add_subplot(grid[i + 1])
        ax.set_facecolor(SURFACE)
        images.append(ax.imshow(np.ones((52, 10)), aspect="auto", origin="lower", cmap=blues, vmin=0.4, vmax=1.8,
                                extent=[-PICTURE, 0, 0, 52], interpolation="nearest"))
        ax.set_ylabel(f"{link.name}\nsubcarrier")
        if i == n - 1:
            ax.set_xlabel("seconds ago (now is at the right edge)")
    t_start = time.time()

    def update(_):
        now = time.time()
        calibrating = now - t_start < CALIBRATE
        parts = []
        for link, line, image in zip(links, lines, images):
            _, a = link.recent(WINDOW, now)
            if a is not None:
                link.level.append((now, a.std(axis=0).mean()))
            if not link.level:
                parts.append(f"{link.name}: no packets")
                continue
            if not calibrating and link.baseline is None:
                link.baseline = max(np.median([v for _, v in link.level]), 1e-6)
            base = link.baseline or np.median([v for _, v in link.level])
            pts = [(t - now, v / base) for t, v in link.level if now - t <= HISTORY]
            line.set_data([p[0] for p in pts], [p[1] for p in pts])
            parts.append(f"{link.name}: {pts[-1][1]:.1f}×")
            t, a = link.recent(PICTURE, now)
            if a is not None:
                # amplitude relative to this receiver's usual level, on an even time grid
                rel = a / np.median(a)
                g = np.linspace(-PICTURE, 0, 300)
                image.set_data(np.stack([np.interp(g, t, rel[:, j], left=np.nan) for j in range(rel.shape[1])]))
                image.set_extent([-PICTURE, 0, 0, rel.shape[1]])
        peak = max([y for line in lines for y in line.get_ydata()] + [3.2])
        top.set_ylim(0, min(peak * 1.25, 40))
        if calibrating:
            status.set_text(f"Measuring the empty room: stay away for {CALIBRATE - (now - t_start):.0f} s")
        else:
            status.set_text("Activity now   " + "    ".join(parts))
        return lines + images + [status]

    anim = FuncAnimation(fig, update, interval=200, cache_frame_data=False)
    plt.show()
    del anim


if __name__ == "__main__":
    main()
