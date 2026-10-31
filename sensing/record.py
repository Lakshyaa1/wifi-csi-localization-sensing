"""
record.py - record labelled CSI clips from one or two ESP32-S3 boards.

Every serial line is saved to a raw log (nothing is thrown away); clip start/end times and labels
go into index.csv. Clips are cut out later by build_dataset.py, so a wrong label can be fixed by
editing index.csv without re-recording.

Examples
    python record.py --ports COM5 COM6 --room A --plan empty:10,walk:10,sit:10,wave:10
    python record.py --ports /dev/ttyUSB0 /dev/ttyUSB1 --room B --plan walk:5 --auto
    python record.py --simulate --room A --plan empty:3,walk:3          # no hardware needed
"""
import argparse, csv, json, os, random, subprocess, sys, threading, time
from collections import deque


class PortReader(threading.Thread):
    """Reads lines from one board forever and appends them to its raw log."""
    def __init__(self, name, source, log_path):
        super().__init__(daemon=True)
        self.name, self.source = name, source
        self.log = open(log_path, "a", buffering=1)
        self.recent = deque()
        self.lock = threading.Lock()
        self.total = 0
        self.last_t = None
        self.max_gap = 0.0      # longest time between packets since reset_gap()
        self.running = True

    def run(self):
        while self.running:
            raw = self.source.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="ignore").strip() if isinstance(raw, bytes) else raw.strip()
            t = time.time()
            self.log.write(f"{t:.6f}\t{line}\n")
            if line.startswith("CSI_DATA"):
                with self.lock:
                    self.total += 1
                    self.recent.append(t)
                    if self.last_t is not None:
                        self.max_gap = max(self.max_gap, t - self.last_t)
                    self.last_t = t

    def reset_gap(self):
        with self.lock:
            self.max_gap = 0.0

    def rate(self, window=1.0):
        now = time.time()
        with self.lock:
            while self.recent and self.recent[0] < now - window:
                self.recent.popleft()
            return len(self.recent) / window


class SimSource:
    """Stand-in for a serial port: streams fake packets in real time (see sim.py)."""
    def __init__(self, room, board, state):
        from sim import SimBoard
        self.sim = SimBoard(room, board, seed=int(time.time()) % 1000)
        self.state, self.t0 = state, time.time()
        self.next_t = 0.0

    def readline(self):
        now = time.time() - self.t0
        if now < self.next_t:
            time.sleep(min(0.01, self.next_t - now))
            return b""
        t = self.next_t
        self.next_t += 0.01 + self.sim.rng.normal(0, 0.001)
        if self.sim.rng.random() < 0.03:          # dropped packet
            return b""
        return self.sim.packet(t, self.state["activity"]).encode()


def parse_plan(s):
    plan = []
    for item in s.split(","):
        lab, n = item.split(":")
        plan.append((lab.strip(), int(n)))
    return plan


def say(text, enabled):
    if enabled:
        subprocess.Popen(["spd-say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def beep():
    sys.stdout.write("\a"); sys.stdout.flush()


class Screen:
    """Full-screen red/green cue window (cue.py) in a child process; does nothing if disabled."""
    def __init__(self, enabled):
        self.p = None
        if enabled:
            cue = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cue.py")
            self.p = subprocess.Popen([sys.executable, cue], stdin=subprocess.PIPE, text=True)

    def show(self, state, label, secs=0.0, info=""):
        if self.p and self.p.poll() is None:
            end = time.time() + secs if secs else 0
            try:
                self.p.stdin.write(f"{state}\t{label}\t{end:.3f}\t{info}\n"); self.p.stdin.flush()
            except (BrokenPipeError, OSError):
                self.p = None

    def close(self):
        if self.p:
            try:
                self.p.stdin.close()
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ports", nargs="*", default=[], help="serial ports, one per board")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--room", required=True, help="room name, e.g. A or lab")
    ap.add_argument("--person", default="p1", help="who is acting (for later analysis)")
    ap.add_argument("--plan", required=True, help="label:count list, e.g. empty:10,walk:10")
    ap.add_argument("--clip-sec", type=float, default=5.0)
    ap.add_argument("--gap-sec", type=float, default=8.0, help="pause between clips in --auto mode")
    ap.add_argument("--auto", action="store_true", help="run clips on a timer instead of pressing Enter")
    ap.add_argument("--no-shuffle", action="store_true", help="keep labels in plan order (not recommended)")
    ap.add_argument("--min-rate", type=float, default=80.0, help="warn below this packet rate (per s)")
    ap.add_argument("--max-gap", type=float, default=0.3,
                    help="flag a clip if any board goes this long (s) without a packet")
    ap.add_argument("--ping", nargs="?", const="auto", default=None,
                    help="ping the router 100x/s from the laptop to keep its reply rate steady "
                         "(router IP, or no value = default gateway)")
    ap.add_argument("--stop-on-flag", action="store_true",
                    help="stop the session at the first clip with few packets or a stall")
    ap.add_argument("--resume", default=None,
                    help="session folder to continue: records only the clips still missing from --plan")
    ap.add_argument("--out", default="data")
    ap.add_argument("--simulate", action="store_true", help="use fake boards instead of serial ports")
    ap.add_argument("--sim-boards", type=int, default=2)
    ap.add_argument("--speak", action="store_true", help="announce labels, go and stop with spd-say")
    ap.add_argument("--screen", action="store_true", help="full-screen cue window: red = wait, green = act")
    a = ap.parse_args()
    screen = Screen(a.screen)

    pinger = None
    if a.ping:
        ip = a.ping
        if ip == "auto":
            route = subprocess.run(["ip", "route"], capture_output=True, text=True).stdout.split()
            ip = route[route.index("via") + 1] if "via" in route else None
        if ip:
            pinger = subprocess.Popen(["ping", "-i", "0.01", "-q", ip],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Pinging {ip} 100x/s from the laptop")
        else:
            print("No default gateway found; running without --ping")

    plan = parse_plan(a.plan)
    done = {}
    if a.resume:   # count clips already recorded fine; flagged ones get recorded again
        for row in csv.DictReader(open(os.path.join(a.resume, "index.csv"))):
            if row["flag"] == "ok":
                done[row["label"]] = done.get(row["label"], 0) + 1
    labels = [lab for lab, n in plan for _ in range(max(0, n - done.get(lab, 0)))]
    if not a.no_shuffle:
        random.shuffle(labels)   # interleave classes so slow drift doesn't line up with one class

    stamp = time.strftime("%Y%m%d_%H%M%S")
    sdir = a.resume or os.path.join(a.out, f"session_{stamp}_room{a.room}" + ("_sim" if a.simulate else ""))
    os.makedirs(sdir, exist_ok=True)

    sim_state = {"activity": "empty"}
    readers = []
    if a.simulate:
        for b in range(a.sim_boards):
            readers.append(PortReader(f"board{b}", SimSource(a.room, b, sim_state),
                                      os.path.join(sdir, f"raw_board{b}.log")))
    else:
        if not a.ports:
            sys.exit("give --ports (or use --simulate)")
        import serial
        for b, p in enumerate(a.ports):
            ser = serial.Serial()
            ser.port, ser.baudrate, ser.timeout = p, a.baud, 0.2
            ser.dtr = ser.rts = False   # opening with these set resets the board
            ser.open()
            readers.append(PortReader(f"board{b}", ser, os.path.join(sdir, f"raw_board{b}.log")))
    for r in readers:
        r.start()

    if not a.resume:
        json.dump(dict(room=a.room, person=a.person, ports=a.ports, baud=a.baud, plan=plan,
                       clip_sec=a.clip_sec, max_gap=a.max_gap, ping=a.ping, simulated=a.simulate,
                       boards=[r.name for r in readers], started=stamp),
                  open(os.path.join(sdir, "meta.json"), "w"), indent=2)
    print(f"{len(labels)} clips to record" + (f" (already have {done})" if a.resume else ""))

    print(f"Session folder: {sdir}")
    print("Warming up (waiting for every board to stream, up to 30 s) ...")
    t_wait = time.time()
    time.sleep(8)   # board resets on port open and needs ~6 s before CSI starts
    while time.time() - t_wait < 30 and any(r.rate(5) < a.min_rate for r in readers):
        time.sleep(1)
    rates = "  ".join(f"{r.name}: {r.rate(5):.0f}/s" for r in readers)
    print(f"Packet rate  {rates}")
    if any(r.rate(5) < a.min_rate for r in readers):   # 5 s average; 1 s is too jumpy
        screen.show("idle", "boards too slow", info=rates)
        screen.close()
        sys.exit(f"Stopping: below {a.min_rate:.0f}/s after 30 s. Check the router, distance, and that the board "
                 "is connected (see README, 'Fixing a low packet rate').")

    idx_path = os.path.join(sdir, "index.csv")
    new = not os.path.exists(idx_path)
    idx = open(idx_path, "a", newline="", buffering=1)
    w = csv.writer(idx)
    if new:
        w.writerow(["clip_id", "label", "room", "person", "t_start", "t_end", "flag"])

    clip_id, i = 0, 0
    if not new:   # resuming: continue the clip numbering
        clip_id = 1 + max(int(r["clip_id"]) for r in csv.DictReader(open(idx_path)))
    first_id = clip_id
    try:
        while i < len(labels):
            lab = labels[i]
            print(f"\nClip {i + 1}/{len(labels)}   next: {lab.upper()}")
            say(f"next, {lab}", a.speak)
            progress = f"clip {i + 1} of {len(labels)}"
            if a.auto:
                screen.show("wait", lab, max(0.0, a.gap_sec - 3) + 3, progress)
                time.sleep(max(0.0, a.gap_sec - 3))
            else:
                screen.show("wait", lab, 0, progress + "  (operator presses Enter)")
                cmd = input("  Enter = start,  s = skip,  q = quit > ").strip().lower()
                if cmd == "q":
                    break
                if cmd == "s":
                    i += 1; continue
                screen.show("wait", lab, 3, progress)
            for c in (3, 2, 1):
                print(f"  {c} ...", end="", flush=True); time.sleep(1)
            beep(); say("go", a.speak); print("  GO")
            screen.show("go", lab, a.clip_sec, progress)
            counts0 = [r.total for r in readers]
            for r in readers:
                r.reset_gap()
            sim_state["activity"] = lab
            t0 = time.time()
            while time.time() - t0 < a.clip_sec:
                time.sleep(0.5)
                print("    " + "  ".join(f"{r.name} {r.rate():.0f}/s" for r in readers), end="\r")
            t1 = time.time()
            sim_state["activity"] = "empty"
            beep(); say("stop", a.speak); print("\n  STOP")
            screen.show("wait", "stop", 0, progress)
            got = [r.total - c0 for r, c0 in zip(readers, counts0)]
            need = a.min_rate * a.clip_sec
            gaps = [r.max_gap for r in readers]
            if not all(g >= need for g in got):
                flag = "low_rate"
            elif any(g > a.max_gap for g in gaps):
                flag = "gap"            # a stall makes an empty clip look like motion
            else:
                flag = "ok"
            w.writerow([clip_id, lab, a.room, a.person, f"{t0:.6f}", f"{t1:.6f}", flag])
            print(f"  saved clip {clip_id} ({', '.join(str(g) for g in got)} packets, "
                  f"longest gap {max(gaps):.2f} s) [{flag}]")
            # flag only, never re-queue: one slow board made the same label repeat endlessly
            clip_id += 1
            i += 1
            if flag != "ok":
                detail = "; ".join(f"{r.name}: {g} packets, longest gap {mg:.2f} s"
                                   for r, g, mg in zip(readers, got, gaps))
                if a.stop_on_flag:
                    print(f"\nPAUSED: clip {clip_id - 1} ({lab}) flagged {flag}. {detail}")
                    print(f"Continue later with:  --resume {sdir}")
                    screen.show("idle", "paused", info=f"{flag} on clip {clip_id - 1}")
                    break
                print("  Few packets or a stall on at least one board; flagged, moving on.")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        for r in readers:
            r.running = False
        time.sleep(0.3)
        idx.close()
        if pinger:
            pinger.terminate()
        screen.show("done", "done", info=f"{clip_id - first_id} clips saved")
        screen.close()
        print(f"\nDone. {clip_id - first_id} clips this run, in {sdir}")
        print(f"Next:  python check_quality.py {sdir}")


if __name__ == "__main__":
    main()
