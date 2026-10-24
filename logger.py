"""Record labelled ESP32 CSI from one or more csi_recv boards over serial.

Usage:
  python logger.py --list                                  # connected boards and their roles
  python logger.py --preset joint --name s01_roomA --speak --delay 20
  python logger.py --phases still:10,wave:10 --name test   # activity:seconds
  python logger.py --phases still:3:20,wave:3:20 --name t  # activity:zone:seconds

Every receiver listed in boards.csv (serial,role with role rx1, rx2, ...) that is
plugged in is recorded as one link; without boards.csv, --links rx1=/dev/ttyACM0
names them by hand. Every packet row carries the transmitter's packet `id`, so the
links are joined on it later (same id = same transmission = same instant).

With a gap (default 5 s with --preset), a "transition" phase is inserted before
each phase and announced as "next, ...", so you are already in place when the
labelled phase starts; transition packets are dropped in preprocessing.
"""
import argparse
import csv
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import serial
from serial.tools import list_ports

from layout import ACTIVITIES, ZONES

ROOT = Path(__file__).parent
# csi_recv (ESP32-S3) header: 24 metadata fields, then the quoted CSI array.
FIELDS = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
          "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
          "noise_floor", "ampdu_cnt", "channel", "secondary_channel",
          "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word"]
SAVED_META = ["id", "rssi", "noise_floor", "local_timestamp", "channel", "secondary_channel",
              "bandwidth", "sig_mode", "mcs", "sig_len", "rx_state", "first_word", "len"]
GAIN_RE = re.compile(r"compensate_gain ([\d.]+), agc_gain (\d+), fft_gain (-?\d+)")

# (activity, zone, seconds, insert a transition gap before this phase)
PRESETS = {
    # ~8 min: empty room, then every activity at every zone, 20 s each
    "joint": [("empty", "none", 30, True)]
             + [(a, z, 20, True) for z in ZONES for a in ACTIVITIES],
    # ~1 min check of the pipeline: empty, then the three activities at zone 1
    "short": [("empty", "none", 10, True)] + [(a, "1", 10, True) for a in ACTIVITIES],
}


def parse_line(line):
    if not line.startswith("CSI_DATA") or '"[' not in line or not line.endswith(']"'):
        return None
    head, arr = line.split('"[', 1)
    meta = head.rstrip(",").split(",")
    if len(meta) != len(FIELDS):
        return None
    try:
        vals = [int(v) for v in arr.rstrip(']"').split(",")]
    except ValueError:
        return None
    if len(vals) != int(meta[FIELDS.index("len")]):
        return None  # truncated or interleaved with a log line
    return dict(zip(FIELDS, meta)), vals


def parse_phases(spec):
    phases = []
    for p in spec.split(","):
        parts = p.split(":")
        if len(parts) == 2:
            phases.append((parts[0], "none", float(parts[1]), True))
        else:
            phases.append((parts[0], parts[1], float(parts[2]), True))
    return phases


def board_roles():
    """serial number -> role, from boards.csv (if present)."""
    path = ROOT / "boards.csv"
    if not path.exists():
        return {}
    return {r["serial"]: r["role"] for r in csv.DictReader(open(path))}


def find_links(spec):
    """link name -> serial port, from --links or from boards.csv + what is plugged in."""
    if spec:
        return dict(item.split("=") for item in spec.split(","))
    roles = board_roles()
    links = {roles[p.serial_number]: p.device for p in list_ports.comports()
             if p.serial_number in roles and roles[p.serial_number].startswith("rx")}
    return dict(sorted(links.items()))


def say(text, enabled):
    if enabled:
        subprocess.Popen(["spd-say", "-w", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def spoken(activity, zone, prev_zone):
    if activity == "empty":
        return "leave the square"
    return activity if zone in ("none", prev_zone) else f"zone {zone}, {activity}"


class Recorder:
    """One reader thread per receiver; rows are labelled with the phase current at arrival."""

    def __init__(self, links, baud, writer):
        self.writer, self.lock = writer, threading.Lock()
        self.phase = None                      # (activity, zone) or None = do not save
        self.kept = dict.fromkeys(links, 0)
        self.dropped = dict.fromkeys(links, 0)
        self.seen = dict.fromkeys(links, 0)    # all valid packets, saved or not
        self.stop = threading.Event()
        self.threads = []
        for link, port in links.items():
            s = serial.Serial()
            s.port, s.baudrate, s.timeout = port, baud, 0.5
            s.dtr = False
            s.rts = False
            s.open()
            t = threading.Thread(target=self.read, args=(link, s), daemon=True)
            t.start()
            self.threads.append(t)

    def read(self, link, s):
        gain = ("", "", "")  # last compensate_gain, agc_gain, fft_gain seen before a packet
        while not self.stop.is_set():
            line = s.readline().decode(errors="ignore").strip()
            m = GAIN_RE.search(line)
            if m:
                gain = m.groups()
                continue
            if not line.startswith("CSI_DATA"):
                continue
            parsed = parse_line(line)
            if parsed is None:
                self.dropped[link] += 1
                continue
            self.seen[link] += 1
            phase = self.phase
            if phase is None:
                continue
            meta, vals = parsed
            with self.lock:
                self.writer.writerow([f"{time.time():.4f}", link, *phase, *(meta[k] for k in SAVED_META),
                                      *gain, " ".join(map(str, vals))])
            self.kept[link] += 1
        s.close()

    def wait_for_packets(self, seconds=2.0, timeout=40.0):
        """Opening a port can reset the board; wait until every link has streamed for `seconds`."""
        t_end = time.time() + timeout
        while time.time() < t_end:
            before = dict(self.seen)
            time.sleep(seconds)
            rate = {k: (self.seen[k] - before[k]) / seconds for k in before}
            if all(r > 20 for r in rate.values()):
                return rate
        sys.exit(f"No steady CSI on every link after {timeout:.0f} s (packets/s: {rate}). "
                 "Is the transmitter powered?")

    def close(self):
        self.stop.set()
        for t in self.threads:
            t.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show connected serial boards and exit")
    ap.add_argument("--links", help="name=port pairs, e.g. rx1=/dev/ttyACM0,rx2=/dev/ttyACM1 "
                    "(default: receivers from boards.csv that are plugged in)")
    ap.add_argument("--baud", type=int, default=2000000)
    ap.add_argument("--phases", help="activity:seconds or activity:zone:seconds, comma separated")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="standard phase list instead of --phases")
    ap.add_argument("--gap", type=float, default=None, help="transition seconds before each phase "
                    "(default 5 with --preset, 0 with --phases)")
    ap.add_argument("--name", help="output file name (no extension)")
    ap.add_argument("--speak", action="store_true", help="announce phases with spd-say")
    ap.add_argument("--delay", type=float, default=0, help="seconds to wait before starting")
    args = ap.parse_args()

    if args.list:
        roles = board_roles()
        for p in list_ports.comports():
            if p.vid:
                print(f"{p.device}  serial={p.serial_number}  role={roles.get(p.serial_number, '(not in boards.csv)')}"
                      f"  {p.product}")
        return
    if not args.name or bool(args.phases) == bool(args.preset):
        ap.error("give --name and exactly one of --phases or --preset")
    links = find_links(args.links)
    if not links:
        ap.error("no receivers found: add them to boards.csv (see --list) or pass --links")
    phases = PRESETS[args.preset] if args.preset else parse_phases(args.phases)
    gap = args.gap if args.gap is not None else (5.0 if args.preset else 0.0)
    schedule, prev_zone = [], None
    for activity, zone, dur, with_gap in phases:
        if gap > 0 and with_gap:
            schedule.append(("transition", "none", gap, f"next, {spoken(activity, zone, prev_zone)}"))
            schedule.append((activity, zone, dur, "go"))
        else:
            schedule.append((activity, zone, dur, spoken(activity, zone, prev_zone)))
        prev_zone = zone
    total = sum(d for _, _, d, _ in schedule)
    out = ROOT / "data" / f"{args.name}.csv"
    out.parent.mkdir(exist_ok=True)
    print(f"links: {links}")
    print(f"{len(phases)} phases, {total / 60:.1f} min total -> {out}")

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["host_time", "link", "activity", "zone", *SAVED_META,
                    "compensate_gain", "agc_gain", "fft_gain", "csi"])
        rec = Recorder(links, args.baud, w)
        rate = rec.wait_for_packets()
        print("packets/s per link: " + ", ".join(f"{k} {v:.0f}" for k, v in rate.items()))
        if args.delay > 0:
            say(f"Recording starts in {int(args.delay)} seconds.", args.speak)
            time.sleep(args.delay)
        for activity, zone, dur, cue in schedule:
            print(f"\n>>> {activity} zone={zone} for {dur:.0f} s", end="", flush=True)
            sys.stdout.write("\a")
            say(cue, args.speak)
            rec.phase = (activity, zone)
            t_end = time.time() + dur
            while time.time() < t_end:
                time.sleep(min(1.0, max(0.0, t_end - time.time())))
                print(f" {max(0, int(round(t_end - time.time())))}", end="", flush=True)
        rec.phase = None
        rec.close()
    say("Recording finished", args.speak)
    print(f"\n\nSaved to {out}")
    for k in links:
        print(f"  {k}: {rec.kept[k]} packets ({rec.dropped[k]} malformed lines dropped)")


if __name__ == "__main__":
    main()
