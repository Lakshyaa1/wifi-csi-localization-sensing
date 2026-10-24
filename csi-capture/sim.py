"""
sim.py - fake ESP32-S3 CSI in the exact serial format of esp-csi `csi_recv_router`.

Use it to practise the whole pipeline before (or without) hardware:
    python sim.py --out data --room A          # writes a full fake session in seconds
    python sim.py --out data --room B          # room B has different geometry -> domain shift
record.py --simulate also uses this module to stream fake packets in real time.

The physics is the multipath sum from Eq. 2 of the MU-SHOT-Fi paper: a few static paths
plus one path that bounces off the person, whose length changes with the activity.
"""
import argparse, os, time, json, csv
import numpy as np

C = 299792458.0
FC = 2.437e9                       # channel 6
DF = 312.5e3                       # subcarrier spacing
K = np.arange(-32, 32)             # L-LTF subcarrier indices, frequency order
NULL = (K == 0) | (np.abs(K) > 26)
ACTIVITIES = ["empty", "walk", "sit", "wave"]

ROOMS = {   # static paths: (length m, amplitude); person path base length
    "A": dict(static=[(4.0, 1.0), (6.5, 0.45), (5.2, 0.3)], person=5.0),
    "B": dict(static=[(3.2, 1.0), (4.9, 0.6), (7.4, 0.35), (4.1, 0.25)], person=4.2),
}


def person_offset(activity, t, phase):
    """Extra path length (m) caused by the person at time t (s)."""
    if activity == "walk":                       # back and forth, path changes up to ~1.6 m/s
        return 0.5 * np.sin(2 * np.pi * 0.25 * t + phase) * 2
    if activity == "sit":                        # one slow down-up movement over the clip
        return 0.35 * (1 - np.cos(2 * np.pi * (t % 3.0) / 3.0)) / 2 * 2
    if activity == "wave":                       # small fast arm motion
        return 0.06 * np.sin(2 * np.pi * 2.2 * t + phase) * 2
    return 0.0


def csi_frame(room, board, activity, t, phase, rng):
    g = ROOMS[room]
    f = FC + K * DF
    H = np.zeros(64, complex)
    for L, a in g["static"]:
        L = L + 0.4 * board                      # each board sits somewhere else
        H += a * np.exp(-2j * np.pi * f * L / C)
    if activity != "empty":
        Lp = g["person"] + 0.4 * board + person_offset(activity, t, phase)
        H += 0.35 * np.exp(-2j * np.pi * f * Lp / C)
    H *= 28.0
    H += rng.normal(0, 0.9, 64) + 1j * rng.normal(0, 0.9, 64)   # receiver noise
    if rng.random() < 0.01:                      # occasional impulsive spike
        H *= 2.5
    H[NULL] = 0
    Hb = np.fft.ifftshift(H)                     # back to buffer order 0..31,-32..-1
    buf = np.empty(128, int)
    buf[0::2] = np.clip(np.round(Hb.imag), -127, 127)   # imaginary first
    buf[1::2] = np.clip(np.round(Hb.real), -127, 127)
    return buf


def format_line(seq, ts_us, rssi, buf, mac="aa:bb:cc:dd:ee:ff", channel=6):
    head = (f"CSI_DATA,{seq},{mac},{rssi},11,0,0,0,0,0,0,0,0,0,-96,0,{channel},0,"
            f"{ts_us & 0xFFFFFFFF},0,64,0,{len(buf)},0")
    return head + ',"[' + ",".join(str(v) for v in buf) + ']"'


class SimBoard:
    """Produces timed packets for one board (100 Hz, jitter, ~3% loss)."""
    def __init__(self, room, board, rate=100, seed=0):
        self.room, self.board, self.rate = room, board, rate
        self.rng = np.random.default_rng(seed + 17 * board)
        self.ts0 = int(self.rng.integers(1e6, 4e9))
        self.drift = 1 + self.rng.normal(0, 2e-5)
        self.seq = 0
        self.phase = self.rng.uniform(0, 2 * np.pi)

    def packet(self, t, activity):
        self.seq += 1
        ts = int(self.ts0 + t * 1e6 * self.drift)
        buf = csi_frame(self.room, self.board, activity, t, self.phase, self.rng)
        rssi = int(-45 - 4 * self.board + self.rng.normal(0, 1))
        return format_line(self.seq, ts, rssi, buf)

    def times(self, t_start, t_end):
        t = np.arange(t_start, t_end, 1 / self.rate)
        t = t + self.rng.normal(0, 0.002, len(t))
        return t[self.rng.random(len(t)) > 0.03]


def generate_session(out, room, plan, clip_sec=3.0, gap_sec=2.0, boards=2, seed=0, person="sim"):
    """Write a complete fake session (raw logs + index.csv) instantly, same layout as record.py."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    sdir = os.path.join(out, f"session_{stamp}_room{room}_sim")
    os.makedirs(sdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    labels = [lab for lab, n in plan for _ in range(n)]
    rng.shuffle(labels)
    host0 = time.time()
    sims = [SimBoard(room, b, seed=seed) for b in range(boards)]
    # timeline: gap, clip, gap, clip ...
    segments, t = [], 0.0
    for i, lab in enumerate(labels):
        segments.append((t, t + gap_sec, "empty", None)); t += gap_sec
        segments.append((t, t + clip_sec, lab, i)); t += clip_sec
    segments.append((t, t + gap_sec, "empty", None)); t_end = t + gap_sec
    for b, sim in enumerate(sims):
        with open(os.path.join(sdir, f"raw_board{b}.log"), "w") as f:
            for tp in sim.times(0, t_end):
                act = next((s[2] for s in segments if s[0] <= tp < s[1]), "empty")
                usb_jitter = abs(rng.normal(0.004, 0.003))
                f.write(f"{host0 + tp + usb_jitter:.6f}\t{sim.packet(tp, act)}\n")
    with open(os.path.join(sdir, "index.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "label", "room", "person", "t_start", "t_end", "flag"])
        for s0, s1, lab, i in segments:
            if i is not None:
                w.writerow([i, lab, room, person, f"{host0 + s0:.6f}", f"{host0 + s1:.6f}", "ok"])
    json.dump(dict(room=room, simulated=True, boards=[f"board{b}" for b in range(boards)],
                   clip_sec=clip_sec, plan=plan), open(os.path.join(sdir, "meta.json"), "w"), indent=2)
    return sdir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Write a fake CSI session for testing the pipeline.")
    ap.add_argument("--out", default="data")
    ap.add_argument("--room", default="A", choices=list(ROOMS))
    ap.add_argument("--clips", type=int, default=20, help="clips per activity")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = generate_session(a.out, a.room, [(x, a.clips) for x in ACTIVITIES], seed=a.seed)
    print("wrote", d)
