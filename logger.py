"""Record labelled ESP32 CSI from the csi_recv example over serial.

Usage:
  python logger.py --phases still:10,wave:10,still:10 --name wave_test
Each phase prints a prompt and a countdown; every CSI packet is saved with
the label of the phase it arrived in.
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import serial

# csi_recv (ESP32-S3) header: 24 metadata fields, then the quoted CSI array.
FIELDS = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
          "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
          "noise_floor", "ampdu_cnt", "channel", "secondary_channel",
          "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word"]


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


def say(text, enabled):
    if enabled:
        subprocess.Popen(["spd-say", "-w", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM1")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--phases", required=True, help="label:seconds,label:seconds,...")
    ap.add_argument("--name", required=True, help="output file name (no extension)")
    ap.add_argument("--speak", action="store_true", help="announce phases with spd-say")
    ap.add_argument("--delay", type=float, default=0, help="seconds to wait before starting")
    args = ap.parse_args()

    phases = [(p.split(":")[0], float(p.split(":")[1])) for p in args.phases.split(",")]
    out = Path(__file__).parent / "data" / f"{args.name}.csv"

    if args.delay > 0:
        say(f"Recording starts in {int(args.delay)} seconds. Get into position.", args.speak)
        time.sleep(max(0, args.delay - 3))
    # Set DTR/RTS before opening: toggling them on open resets the ESP32-S3,
    # which costs ~8 s of reboot + Wi-Fi init at the start of the first phase.
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = args.port, args.baud, 0.5
    s.dtr = False
    s.rts = False
    s.open()
    print("Warming up 3 s (stay where you are)...")
    time.sleep(3)
    s.reset_input_buffer()

    kept = dropped = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["host_time", "label", "rssi", "noise_floor", "local_timestamp", "len", "csi"])
        for label, dur in phases:
            print(f"\n>>> {label.upper()} for {dur:.0f} s", end="", flush=True)
            sys.stdout.write("\a")
            say(f"{label} now", args.speak)
            t_end = time.time() + dur
            next_tick = time.time() + 1
            while time.time() < t_end:
                line = s.readline().decode(errors="ignore").strip()
                if time.time() >= next_tick:
                    print(f" {int(t_end - time.time())}", end="", flush=True)
                    next_tick += 1
                if not line.startswith("CSI_DATA"):
                    continue
                parsed = parse_line(line)
                if parsed is None:
                    dropped += 1
                    continue
                meta, vals = parsed
                w.writerow([f"{time.time():.4f}", label, meta["rssi"], meta["noise_floor"],
                            meta["local_timestamp"], meta["len"], " ".join(map(str, vals))])
                kept += 1
    s.close()
    say("Recording finished", args.speak)
    print(f"\n\nSaved {kept} packets to {out} ({dropped} malformed lines dropped)")


if __name__ == "__main__":
    main()
