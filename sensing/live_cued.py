"""
live_cued.py - a scored live run: the cue screen tells the actor what to do while the CNN guesses live.

    python live_cued.py dataset.npz --ports <board0> <board1> --room Hlive2 --script empty:60,walk:15,...

The CNN is learned on every clip in dataset.npz (5 seeds, averaged). Every half second the last 5 s from both
boards is cleaned as in build_dataset.py and classified two ways, on the same window, while it happens:
  none   the model as learned
  blend  batch-norm averages = half the learned ones, half those of every window seen so far in this run
         (refreshed every 10 windows, no labels; the first 10 windows use the unadapted model)
  spec_none, spec_blend  the same two, for a CNN whose input is the scale-free ripple-rate picture
         (spec_check.py, 'spec')
The blend sees every window, including the pauses between blocks. A guess is scored only when its whole 5 s
lies inside one cued block (the cue is the label; it is never given to the model).
Saved in the session folder: raw_boardN.log and index.csv (same format as record.py, so the run can be
replayed), guesses.csv (every guess) and summary.txt.
"""
import argparse, csv, json, os, threading, time
import numpy as np
import live
from record import Screen, SimSource, parse_plan


class LogBoard(live.Board):
    """live.Board that takes an open source (serial port or simulator) and also writes the raw log."""
    def __init__(self, source, log_path, keep_sec=8.0):
        threading.Thread.__init__(self, daemon=True)
        import collections
        self.ser, self.log = source, open(log_path, "a", buffering=1)
        self.buf, self.keep_sec, self.first_t, self.lock = collections.deque(), keep_sec, None, threading.Lock()

    def run(self):
        while True:
            raw = self.ser.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="ignore").strip()
            t = time.time()
            self.log.write(f"{t:.6f}\t{line}\n")
            p = live.parse_line(line)
            if p is None or len(p["csi"]) != 128:
                continue
            with self.lock:
                if self.first_t is None:
                    self.first_t = t
                self.buf.append((t, p["ts_us"], p["first_word_invalid"], p["csi"]))
                while self.buf and self.buf[0][0] < t - self.keep_sec:
                    self.buf.popleft()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--ports", nargs="*", default=[])
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--room", required=True)
    ap.add_argument("--script", required=True, help="label:seconds blocks in order, e.g. empty:60,walk:15")
    ap.add_argument("--gap-sec", type=float, default=8.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--warmup", type=float, default=90.0)
    ap.add_argument("--out", default="data")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--no-screen", action="store_true")
    a = ap.parse_args()

    d = np.load(a.dataset)
    labels, fs = [str(l) for l in d["labels"]], int(d["fs"])
    clip_sec = d["X"].shape[-1] / fs
    blocks = [(lab, float(n)) for lab, n in parse_plan(a.script)]
    print(f"training the CNN on all {len(d['X'])} clips, {a.seeds} seeds ...")
    import torch
    from train_cnn import fit, prepare
    X, y = prepare(d["X"]), torch.from_numpy(d["y"].astype(np.int64))
    sources = [fit(X, y, len(labels), seed) for seed in range(a.seeds)]
    from spec_check import spec
    tf = lambda x: spec(x, False)
    spec_sources = [fit(tf(X), y, len(labels), seed) for seed in range(a.seeds)]
    predict = {"none": live.train_cnn_stream(a.dataset, a.seeds, False, sources),
               "blend": live.train_cnn_stream(a.dataset, a.seeds, True, sources),
               "spec_none": live.train_cnn_stream(a.dataset, a.seeds, False, spec_sources, tf),
               "spec_blend": live.train_cnn_stream(a.dataset, a.seeds, True, spec_sources, tf)}
    names = list(predict)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    sdir = os.path.join(a.out, f"session_{stamp}_room{a.room}" + ("_sim" if a.simulate else ""))
    os.makedirs(sdir)
    sim_state = {"activity": "empty"}
    boards = []
    for b in range(2 if a.simulate else len(a.ports)):
        if a.simulate:
            src = SimSource(a.room, b, sim_state)
        else:
            import serial
            src = serial.Serial()
            src.port, src.baudrate, src.timeout = a.ports[b], a.baud, 0.2
            src.dtr = src.rts = False
            src.open()
        boards.append(LogBoard(src, os.path.join(sdir, f"raw_board{b}.log")))
    for b in boards:
        b.start()
    json.dump(dict(room=a.room, ports=a.ports, script=blocks, gap_sec=a.gap_sec, seeds=a.seeds, dataset=a.dataset,
                   started=stamp, kind="live_cued"), open(os.path.join(sdir, "meta.json"), "w"), indent=2)
    print(f"Session folder: {sdir}")

    screen = Screen(not a.no_screen)
    cur = {"label": None, "t0": 0.0, "t1": 0.0, "last": "", "run": True}
    rows = []   # (t, cued label or "", guess none, guess blend)
    gfile = open(os.path.join(sdir, "guesses.csv"), "w", newline="", buffering=1)
    gw = csv.writer(gfile)
    gw.writerow(["t", "cued", "scored"] + names + [f"p_{n}_{l}" for n in names for l in labels])

    def guess_loop():
        nxt = time.time()
        while cur["run"]:
            nxt += 0.5
            time.sleep(max(0.0, nxt - time.time()))
            t1 = time.time() - 0.3
            clip, why = live.window_clip(boards, t1, clip_sec, fs)
            if clip is None:
                cur["last"] = why
                continue
            ps = [predict[n](clip) for n in names]
            lab = cur["label"]
            scored = lab in labels and t1 - clip_sec >= cur["t0"] and t1 <= cur["t1"]
            gs = [labels[int(p.argmax())] for p in ps]
            rows.append((t1, lab if scored else "", *gs))
            gw.writerow([f"{t1:.3f}", lab or "", int(scored)] + gs + [f"{q:.3f}" for p in ps for q in p])
            cur["last"] = f"signal: {gs[0]} / blend {gs[1]}     ripple-rate: {gs[2]} / blend {gs[3]}"

    print(f"Waiting for both boards to stream (up to {a.warmup:.0f} s) ...")
    screen.show("idle", "warming up")
    t_wait = time.time()
    while time.time() - t_wait < a.warmup and live.window_clip(boards, time.time() - 0.3, clip_sec, fs)[0] is None:
        time.sleep(1)
    if live.window_clip(boards, time.time() - 0.3, clip_sec, fs)[0] is None:
        screen.show("idle", "boards not streaming"); screen.close()
        raise SystemExit("Stopping: no clean 5 s window from both boards after the warm-up.")
    threading.Thread(target=guess_loop, daemon=True).start()

    idx = open(os.path.join(sdir, "index.csv"), "w", newline="", buffering=1)
    w = csv.writer(idx)
    w.writerow(["clip_id", "label", "room", "person", "t_start", "t_end", "flag"])
    try:
        for i, (lab, dur) in enumerate(blocks):
            prog = f"block {i + 1} of {len(blocks)}"
            print(f"{prog}: next {lab} for {dur:.0f} s", flush=True)
            end = time.time() + a.gap_sec
            while time.time() < end:
                screen.show("wait", lab, end - time.time(), f"{prog}   {cur['last']}")
                time.sleep(0.5)
            t0 = time.time()
            cur.update(label=lab, t0=t0, t1=t0 + dur)
            sim_state["activity"] = lab
            while time.time() < t0 + dur:
                screen.show("go", lab, t0 + dur - time.time(), f"{prog}   {cur['last']}")
                time.sleep(0.5)
            t1 = time.time()
            sim_state["activity"] = "empty"
            w.writerow([i, lab, a.room, "p1", f"{t0:.6f}", f"{t1:.6f}", "ok"])
            time.sleep(0.4)   # let the last guess of the block land before the label is cleared
            cur.update(label=None)
    except KeyboardInterrupt:
        print("\nStopped early.")
    cur["run"] = False
    time.sleep(0.7)
    screen.show("done", "done"); screen.close()

    out = [f"{len(rows)} guesses, {sum(1 for r in rows if r[1])} scored (whole 5 s inside one block)"]
    sc = [r for r in rows if r[1]]
    truth = np.array([r[1] for r in sc])
    for j, name in enumerate(names, start=2):
        g = np.array([r[j] for r in sc])
        if not len(sc):
            break
        per = {l: (g[truth == l] == l).mean() for l in labels if (truth == l).any()}
        out.append(f"  {name:10} right {np.mean(g == truth):6.1%}   per-activity avg {np.mean(list(per.values())):6.1%}   "
                   + "  ".join(f"{l} {v:.0%} ({(truth == l).sum()})" for l, v in per.items()))
    print("\n".join(out))
    open(os.path.join(sdir, "summary.txt"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
