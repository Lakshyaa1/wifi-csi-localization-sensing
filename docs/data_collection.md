# Data collection guide

Step-by-step instructions for recording a labelled dataset with one router and two ESP32-S3
boards. The first goal is a feasibility check:

> Train on one session, test on a second session recorded later. Is accuracy clearly above
> chance (25% for 4 classes), and is `empty` vs moving above about 80%?

Hardware steps below were tested on real boards on Oct 24, 2026, except where marked.

## 1. What you need

- A 2.4 GHz router (a normal home router is fine), 2 × ESP32-S3-DevKitC, 2 USB **data** cables
  (charge-only cables power the board but the laptop sees nothing), one laptop with Python 3.
- Three people: operator (laptop, sits still outside the area), actor, checker (films the
  session on a phone and checks the actor does the right action).

## 2. Router

- Use the 2.4 GHz network, with a **fixed channel** (not "Auto") and 20 MHz width.
- Ideally nobody else uses that network while recording.

## 3. Flash both boards (once)

ESP-IDF v5.x. If you use conda, take it off `PATH` before sourcing `export.sh`.

```bash
git clone https://github.com/espressif/esp-csi
cd esp-csi
git apply /path/to/this/repo/firmware/csi_recv_router.patch   # gain lock + power save off
cd examples/get-started/csi_recv_router
idf.py set-target esp32s3
idf.py menuconfig      # Example Connection Configuration -> SSID and password (case-sensitive!)
idf.py -p /dev/ttyACM0 flash monitor
```

Good: within about 6 s you see `got ip`, then a stream of lines starting with `CSI_DATA,`.
Exit with `Ctrl-]`. Flash the second board the same way.

Both changes in the patch matter:
- **Power save off.** Without it the rate collapses to a few packets per second after the
  first seconds. This cost us a whole afternoon.
- **Gain lock** (`CONFIG_FORCE_GAIN 1`). Otherwise the automatic gain makes the amplitude jump
  on its own, which looks like motion.

If the monitor shows `Wi-Fi disconnected 201`, the board can't find the network: check the
SSID spelling and case, and that it is the 2.4 GHz network.

## 4. Python

```bash
cd sensing
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib scikit-learn pyserial
python sim.py --room A --out data && python check_quality.py data/session_*_roomA_sim   # fake data, checks the code runs
```

Find the ports with `ls /dev/serial/by-id/` (Linux) or Device Manager (Windows). Use the
`/dev/serial/by-id/...` names on Linux: they don't change when you replug.

## 5. Placement

```
  router                                   board 0
     ●  ─────────────── 3 m ─────────────── ●   ↑
       ╲───────╲                               │ 1 m apart
            ╲──────── ✕ ─────────────────── ●   ↓
                     spot                  board 1
                  (1.5 m from router)
```

- Router 3 m from each board; boards about 1 m apart; everything about 1 m off the floor
  (on tables), clear line of sight.
- The activity spot is halfway along, between the two lines. Put a chair there for `sit`.
- Why: the region where a body changes the signal is only about 30 cm either side of the
  straight router–board line (the first Fresnel zone). Standing beside the line instead of on
  it was why an earlier attempt failed.
- **Tape everything down and don't touch the boards or cables during a session.** Moving a
  board 10 cm changes the CSI more than the activities do.
- Measure and photograph the layout.

## 6. Record

Two sessions, at least an hour apart (or on two days), in the same room with the same layout.
Name them as different "rooms" so the training script can use one to train and the other to test:

```bash
python record.py --ports <port0> <port1> --room A1 --plan empty:10,walk:10,sit:10,wave:10 --auto --screen --ping --stop-on-flag
# later
python record.py --ports <port0> <port1> --room A2 --plan empty:10,walk:10,sit:10,wave:10 --auto --screen --ping --stop-on-flag
```

- Clips are 5 s with 8 s gaps (`--clip-sec`, `--gap-sec`): 40 clips take about 9 minutes.
  Labels come in random order.
- `--ping` makes the laptop ping the router 100 times a second during the session, which keeps
  the router's reply rate steady (it finds the router itself; or give its IP).
- `--speak` reads the cues aloud (Linux, `spd-say`): "next, walk", then a 3 s count, "go",
  "stop". Without `--auto`, the operator presses Enter for each clip.
- `--screen` opens a full-screen cue window the actor can see: red = wait (next action and a
  countdown to GO), green = do the action (countdown to STOP). Esc closes it; recording goes on.
- The script stops at the start if the boards don't average 80/s or more over 5 s (`--min-rate`).
- **Before a full session, run a 1-minute check:** 30 s with the actor outside the room, then
  30 s walking across the lines. Only record the session if walking clearly changes the signal
  more than the empty room does.
- Actions:
  - `empty`: actor leaves the room (in a small room, standing behind the boards still moves the signal).
  - `walk`: walk back and forth across both lines, within about 1 m of the spot.
  - `sit`: one sit-down-and-stand-up on the chair at the spot.
  - `wave`: stand on the spot and wave one arm. Only wave. Don't walk.
- Wrong action in a clip? Write it down and change that clip's `flag` to `bad_label` in the
  session's `index.csv`. The raw data is never cut, so nothing is lost.
- A clip with too few packets is flagged `low_rate`; a clip where any board goes more than
  0.3 s without a packet is flagged `gap` (a stall makes an empty clip look like motion).
  `build_dataset.py` leaves flagged clips out. With `--stop-on-flag` the session stops right
  after a flagged clip; continue it later with `--resume <session folder>`, which records only
  the clips still missing.

## 7. Check and train

```bash
python check_quality.py data/session_..._roomA1      # after each session
python build_dataset.py data/session_..._roomA1 data/session_..._roomA2 --out dataset.npz --plots figs
python train_baseline.py dataset.npz --source A1 --target A2
```

Good quality looks like: 90–100 packets/s on both boards, "52 carry signal", RSSI steady, and in
`quality.png` the heatmap changes during motion clips and stays calm during `empty`.

To see live guesses on screen (learned from a dataset, last 5 s, updated every 0.5 s; give the
ports in the same order as when recording):

```bash
python live.py dataset.npz --ports <port0> <port1>
```

`train_baseline.py` prints two numbers. The **"new room (A2)"** accuracy is the one that matters
here: trained on session A1, tested on session A2. Also look at `figs/classes.png` to see what
each activity looks like.

## 8. Problems we already hit

| Symptom | Cause / fix |
|---|---|
| Laptop sees no board | Charge-only cable, or a loose plug. Try another cable. |
| `Wi-Fi disconnected 201` in the monitor | Wrong SSID (it is case-sensitive) or the 5 GHz network. |
| Rate starts at 100/s, then drops to a few per second | Power save is on: apply the patch and reflash. |
| One board loses Wi-Fi mid-session ("beacon timeout") | Seen once on Oct 24, at strong signal. Probably power or cable; not confirmed. Try another cable or USB port. |
| Laptop can't power two boards | Use a powered USB hub, or a second laptop for the second board. |
| `empty` clips look as noisy as motion clips | Seen on Oct 24 and Oct 29 with a far router (about 25% change within a clip, as much as walking). On Oct 31, in a small room with a nearby router and the actor outside the room, empty clips changed about 5–11% (median per board and session) and every action clip in session 1 scored higher. The far router's packets came in two alternating shapes; the cause is not confirmed. |
| A board suddenly gives about 0 packets/s after a restart | Two routers broadcast the same network name; after a restart the board joined the far one (RSSI −94). Fix: in `sdkconfig` set `CONFIG_EXAMPLE_WIFI_SCAN_RSSI_THRESHOLD=-70` and reflash, so weak routers are ignored. |
| Rate fine at first, then varies or collapses on a shared router | The router's reply rate depends on network load. On Oct 29 it held 100/s only while the laptop also pinged the router (`ping -i 0.01 <gateway>`). Packets also dropped when a person stood on the link. |

## 9. What to send back

Zip each `data/session_...` folder (raw logs, `index.csv`, `meta.json`, `quality.png`), plus the
`train_baseline.py` output, `figs/`, the layout photo and distances, and notes on anything that
went wrong.
