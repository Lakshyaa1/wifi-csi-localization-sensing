# CSI capture: router → ESP32-S3

Record Wi-Fi CSI from a router with ESP32-S3 boards, clean it, and build a labelled dataset for
activity recognition across rooms.

| file | what it does |
|---|---|
| `record.py` | records labelled clips from 1–2 boards (or fake boards with `--simulate`) |
| `check_quality.py` | checks a session right after recording; saves `quality.png` |
| `build_dataset.py` | cuts clips, cleans them, saves `dataset.npz`; `--plots` makes the DSP figures |
| `train_baseline.py` | trains in room A, tests in room A and room B, shows the drop |
| `csi_common.py` | parsing and the DSP steps (shared by everything above) |
| `sim.py` | fake CSI in the exact firmware format, for practice without hardware |

Python: `pip install numpy scipy matplotlib scikit-learn pyserial`

---

## 1. Try the whole pipeline now, without hardware

```
python sim.py --out data --room A
python sim.py --out data --room B
python check_quality.py data/session_..._roomA_sim
python build_dataset.py data/session_*_roomA_sim data/session_*_roomB_sim --out dataset.npz --plots figs
python train_baseline.py dataset.npz --source A --target B
python record.py --simulate --room A --plan empty:2,walk:2      # practise the recording flow
```

The fake data is much easier than real data, so the accuracy it gives means nothing. It's only
there to check that every step runs.

## 2. Hardware setup

**Router (transmitter)**
- 2.4 GHz only. The ESP32-S3 has no 5 GHz. If the router merges both bands under one name,
  make a separate 2.4 GHz network.
- Fix the channel (e.g. 6) and set the width to 20 MHz. Automatic channel switching ruins a session.
- Keep other heavy traffic off it while recording.

**Boards (receivers)**
- Tape the router and both boards in place, about 1 m off the floor, with the activity area
  between them (2–4 m apart). Mark positions on the floor so room setups can be repeated.
- Photograph the layout and note the distances for each room. You'll need them in the write-up.
- On the ESP32-S3-DevKitC, use the USB port labelled **UART** (the USB-to-serial chip). It
  handles 921600 baud reliably. The other socket (native USB) also works: the firmware copies
  its output there, and we recorded at 100/s through it.

## 3. Firmware (Espressif's esp-csi example)

Needs ESP-IDF (v5.x recommended).

```
git clone https://github.com/espressif/esp-csi
cd esp-csi/examples/get-started/csi_recv_router
idf.py set-target esp32s3
idf.py menuconfig        # Example Connection Configuration -> your 2.4 GHz SSID and password
idf.py -p <PORT> flash monitor
```

You should see lines starting with `CSI_DATA,`. Exit the monitor with `Ctrl-]` before running
`record.py`, since only one program can hold the port.

What the example does: it connects to the router as a normal client and pings the router 100
times a second. The router's replies are the packets we measure, so no laptop traffic is needed.

Easiest: apply our patch, which makes both changes below (gain lock and power save off):

```
cd esp-csi
git apply /path/to/this/repo/csi-capture/firmware/csi_recv_router.patch
```

**Power save must be off.** The unmodified example leaves Wi-Fi power save on. The board then
dozes between beacons, the router's ping replies are delayed or lost, and the CSI rate collapses
from 100/s to a few per second after the first seconds. The patch adds
`esp_wifi_set_ps(WIFI_PS_NONE)` right after `example_connect()`.

Two settings near the top of `main/app_main.c`:
- `CONFIG_SEND_FREQUENCY 100`: pings per second. Leave at 100.
- `CONFIG_FORCE_GAIN 0`: change to **1**. After the first 100 packets this locks the receiver
  gain. Otherwise the automatic gain control makes the amplitude jump on its own, which looks
  like motion.

Flash both boards the same way.

## 4. Recording protocol (team of three)

Roles: **operator** at the laptop (sits still, outside the activity area), **actor**, and a
**checker** who films each session on a phone and watches that the actor does the right thing.

```
python record.py --ports COM5 COM6 --room A --person p1 --plan empty:10,walk:10,sit:10,wave:10
```

- Labels are shuffled, so the slow drift over a session doesn't line up with one class.
- Before each clip the operator presses Enter; the script counts 3-2-1, beeps "GO", records
  3 seconds, beeps "STOP".
- The actor starts at GO and keeps going until STOP. For `sit`, do one sit-down-and-stand-up
  per clip. For `empty`, the actor leaves the area.
- Nobody else moves in the room during a clip.
- Vary the actor's position slightly within the area between clips.
- A clip with too few packets is flagged and automatically re-queued.
- Target: about 50 clips per class per room, over at least two sessions.
- Wrong label? Fix it in that session's `index.csv`. The raw data is never cut, so nothing is lost.

## 5. After every session

```
python check_quality.py data/session_...
```

Good looks like:
- mean rate 90–100 per second on both boards
- almost no gaps over 50 ms
- "52 carry signal" (the 52 data subcarriers of a 20 MHz channel)
- in `quality.png`, the amplitude heatmap visibly changes inside the shaded motion clips and
  stays calm in `empty` clips

**Fixing a low packet rate:** move the boards closer to the router, make sure the band is
2.4 GHz with a fixed channel, try another USB cable or port, test with one board at a time, and
check that `idf.py monitor` shows lines streaming smoothly.

## 6. Build the dataset and the DSP figures

```
python build_dataset.py data/session_A1 data/session_A2 data/session_B1 --out dataset.npz --plots figs
```

Cleaning steps, per clip and per board:
1. CSI integers → complex numbers → amplitude, subcarriers in frequency order
2. keep the 52 data subcarriers (drop DC and the guard bands)
3. drop packets where every subcarrier jumps together (gain glitches)
4. resample onto an exact 100 Hz grid (packets arrive unevenly and some are lost)
5. Hampel filter: replace remaining single-subcarrier spikes with the local median
6. 20 Hz low-pass (human motion at 2.4 GHz is slower than that)

Output: `X` with shape (clips, boards, 52, 300), plus labels and room for each clip.
`figs/dsp_raw_vs_clean.png` and `figs/classes.png` are made for the DSP demo.

## 7. The first experiment

```
python train_baseline.py dataset.npz --source A --target B
```

Trains on room A only and reports accuracy on held-out room-A clips and on room B. The gap
between them is the domain shift. That's the result the adaptation methods from the MU-SHOT-Fi
paper then try to recover.

## Check on your first real session

- Each subcarrier is two signed bytes, **imaginary first**, then real. This is stated in the
  esp-csi README and matches Espressif's own parser, so it should be right.
- The buffer order is subcarriers 0..31 then −32..−1 (Espressif's L-LTF layout; the code
  reorders them). **Checked on real `csi_recv_router` data (Oct 24):** the zeros sit at buffer
  positions 0 (DC) and 27–37 (guard band), as this layout predicts, and `check_quality.py`
  reports 52 subcarriers carrying signal. (The older ESP-NOW `csi_recv` example uses a different
  order, so don't reuse this parser for that firmware.)
