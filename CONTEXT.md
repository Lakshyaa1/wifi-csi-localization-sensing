# Project context: joint sensing and localization with ESP32-S3 Wi-Fi CSI

Status as of 2026-10-12. This note summarises what has been decided, built and learned so far, for someone joining the work.

## 1. What we are building

One model that tells **what a person is doing and where they are**, using only Wi-Fi channel state information (CSI) between ESP32-S3 boards. The person carries nothing. The model should then **adapt to a new room using unlabelled recordings from that room only**.

The adaptation setting is source-free domain adaptation, as in the MU-SHOT-Fi paper (multi-user Wi-Fi sensing on the WiMANS dataset): train with labels in a source room, then adapt in a target room without target labels and without the source data.

### The key design choice
Position fingerprints normally belong to one room, so there is nothing to adapt: "spot A in room 1" has no counterpart in room 2. We define zones **relative to the boards** instead:

- 4 boards at the corners of a 3 m × 3 m square, the same in every room.
- 6 zones inside the square, numbered the same way in every room.

"Zone 4" then means the same thing wherever the square is set up, so label-free adaptation can be applied to the position output as well as the activity output.

What we could claim if it works: a real ESP32 dataset with activity and position labelled together in two rooms, a joint model, and source-free adaptation of both outputs. We would not be claiming a new adaptation algorithm. **Whether the board-relative zone idea is new has not been checked against the literature yet.**

## 2. Setup

### Hardware
- ESP32-S3 dev boards (two USB-C sockets each; the UART socket shows up as a CH343 "USB Single Serial" device).
- Target layout: 1 transmitter + 3 receivers at the corners of the square, about 1 m high, clear line of sight between all boards.

| Board | (x, y) in m |
|---|---|
| tx | (0, 0) |
| rx1 | (3, 0) |
| rx2 | (3, 3) |
| rx3 | (0, 3) |

| Zone | (x, y) | Zone | (x, y) |
|---|---|---|---|
| 1 | (0.75, 1.0) | 4 | (0.75, 2.0) |
| 2 | (1.5, 1.0) | 5 | (1.5, 2.0) |
| 3 | (2.25, 1.0) | 6 | (2.25, 2.0) |

### Boards identified so far
| Role | Wi-Fi MAC | USB serial (UART socket) | Firmware |
|---|---|---|---|
| tx | `b8:1f:3f:bb:02:fc` | `5CE5198291` | `csi_send` |
| rx1 | `7c:e8:b1:b0:50:04` | `5CE5199687` | `csi_recv` |
| rx2 | `a4:cb:8f:d8:92:70` | `5C4C169816` | `csi_recv` |
| rx3 | not yet seen | | |

`boards.csv` maps USB serial numbers to roles, so port names (`/dev/ttyACM0` …) do not matter.

### Firmware
- Espressif's [esp-csi](https://github.com/espressif/esp-csi) `examples/get-started/csi_send` and `csi_recv`, built with ESP-IDF v5.5 for `esp32s3`.
- Link: ESP-NOW broadcast on Wi-Fi channel 11. The transmitter sends about 91 packets/s; a receiver delivers about 70/s to the laptop.
- Each packet carries a counter from the transmitter. Every receiver prints it as `id`, so recordings from several receivers are joined on `id` (same id = same transmission = same instant). No clock synchronisation is needed.
- The receiver prints one `CSI_DATA` line per packet at **921600 baud**. The transmitter's console is at **115200 baud**.
- One local change to `csi_send`: an attempt to blink the on-board RGB LED (GPIO 48 and 38) to mark the transmitter. It does not show on our boards, so boards are marked with tape instead. The change is harmless and is not in any repo.

Flashing (conda must not be on `PATH` when sourcing the IDF environment):

```bash
. ~/esp/esp-idf/export.sh
cd esp-csi/examples/get-started/csi_send && idf.py -p <port> flash   # transmitter
cd ../csi_recv && idf.py -p <port> flash                             # each receiver
```

## 3. Data collection

Activities, all done on the zone's spot so position stays defined: `still`, `walk` (on the spot), `jump` (on the spot), plus `empty` (nobody in the square).

One session is every activity at every zone, 20 s each, with a 5 s cued transition before each block: about 8 minutes.

```bash
python logger.py --list                                   # connected boards and roles
python logger.py --preset joint --name s01_roomA --speak --delay 20
python logger.py --preset short --name test_short --speak # 1 min check
python preprocess.py s01_roomA
```

- Room A (source): 4 sessions over at least 2 days; sessions 1–3 train, session 4 tests.
- Room B (target): 3 sessions; sessions 1–2 are used without labels for adaptation, session 3 tests.
- Full protocol and rules: `protocol.md`.

## 4. Code

| File | Purpose | Status |
|---|---|---|
| `logger.py` | Records several receivers at once (one thread per port); labels each packet with activity and zone; waits for steady packets before starting | Works on one real link; several receivers at once not yet tried |
| `preprocess.py` | Joins receivers on packet id; Hampel filter; resample to 70 Hz; 20 Hz zero-phase Butterworth low-pass; per-subcarrier z-score clipped to ±5; 2 s windows with 50% overlap | Works on one real link; the multi-receiver join is tested on synthetic data only |
| `features.py`, `train_classical.py` | Handcrafted features; Random Forest and SVM for activity or zone; majority-class and shuffled-label checks; zone error in metres | Tested on synthetic data only |
| `layout.py` | Board and zone coordinates, activity list | |
| `phase_check.py`, `plot_csi.py`, `static_vs_motion.py`, `analyze_csi.py` | Earlier analysis and plotting scripts | |
| `dataset.py`, `models.py`, `train.py` | Joint CNN with activity and zone outputs | Not written (week 2) |
| `adapt.py` | Adaptation to room B | Not written (week 3) |

Preprocessed format, `data/processed/<name>.npz`: `X [windows, links × 2, 140, 52]` (per link: amplitude, then sanitized phase), with `activity`, `zone`, `t0`, `session`, `room`, `links`, `subcarriers`.

CSI details:
- Each packet has 384 values = 192 complex numbers (legacy LTF 64 + HT-LTF 128), **imaginary part first**.
- Only the 64-subcarrier legacy LTF is used for now; its order is 0..31, −32..−1, and 52 subcarriers are active.
- Raw phase is random from packet to packet. It is cleaned by removing the linear trend across subcarriers per packet.

Environment: a `.venv` in the repo with numpy, scipy, scikit-learn, torch (CPU), matplotlib and pyserial.

**Where the code is:** the public repo `Lakshyaa1/wifi-csi-localization-sensing` and `pushkardube/wifi_csi` hold only the first motion-detection scripts. Everything in the table marked as new (multi-receiver logger, preprocessing, baselines, layout, protocol) exists only on Lakshya's laptop and has not been committed or pushed.

## 5. Results so far

**First experiment (two boards, 80 cm apart, amplitude only):** a single threshold on a motion score separated still from motion in 82% of 0.5 s windows. At that range, motion mostly blocked the direct path, so all subcarriers dimmed together and RSSI reacted too. Most errors were in the first seconds of each motion phase, where the label led the actual movement; the transition gaps in the new logger address this.

**Sanitized phase:** fluctuates about twice as much during motion as when still.

**Frequency content (one subcarrier, same recording):** motion energy is below 5 Hz; above 5 Hz the still and motion phases look alike, so that band is mostly noise. Low-pass filtering reduced jitter but did not change the still-to-motion separation, which suggests a lower cutoff or a high-pass (background removal) is worth testing.

**Link check on the current boards:** about 70 packets/s received of 91 sent (70–78%), no malformed lines, signal strength between −40 and −68 dBm depending on placement.

No labelled session with the new protocol has been recorded yet, and no model has been trained on real data.

## 6. Plan (4 weeks from Oct 10)

| Week | Dates | Work |
|---|---|---|
| 1 | Oct 10–17 | All boards recording; room A sessions 1–2; classical baseline; literature check |
| 2 | Oct 18–24 | Room A sessions 3–4; joint CNN; 1 receiver vs 3; separate vs joint training |
| 3 | Oct 25–31 | Room B sessions; measure the cross-room drop; adaptation ladder (no adaptation, AdaBN, information maximisation, plus a self-supervised task), 3 seeds |
| 4 | Nov 1–7 | Ablations (background removal, Doppler spectrogram input, tomography baseline); write-up |

Decision points:
- End of week 2: if 3 receivers do not beat 1 for zones, change the layout before recording room B.
- End of week 3: a clear drop in room B and a recovery that holds over 3 seeds makes a workshop or work-in-progress paper worth writing, after adding a second person.

Week 1 is behind: the 4-board setup is not ready.

## 7. Evaluation rules
- Split train and test by session, never by window (neighbouring windows overlap by 50%).
- Room B labels are used only for the final evaluation, never for adaptation or model selection.
- Report last-epoch results over 3 seeds, mean and standard deviation.
- Always include a majority-class baseline and a shuffled-label check.

## 8. Practical problems we hit
- **Charge-only USB cables:** the board powers up but the laptop sees nothing. This happened several times.
- **USB power or port limits:** the laptop has only handled one board at a time; a second caused disconnects and protocol errors. Three receivers will need a powered USB hub or a second laptop.
- **Opening the serial port resets the board,** costing the first seconds of a recording. The logger now waits until every link delivers steady packets before starting.
- **A board has a different serial number on each of its two USB sockets.** Use the same socket every time.
- **Two transmitters must never be powered together:** both use the same sender address, so packet ids would mix.
- **Different console speeds** for transmitter (115200) and receiver (921600) firmware.
- The transmitter's packet counter restarts when it is powered on, so it must stay powered for a whole recording.

## 9. Ideas considered and set aside
| Idea | Decision |
|---|---|
| Federated learning across rooms | Dropped. It needs labels in every room and does not fix the drop in a new unlabelled room; adaptation does. |
| Two-way ranging or Wi-Fi FTM | Locates a carried device, not a device-free person. Out of scope. |
| Angle of arrival from two ESP32s sharing a crystal | Needs fine soldering and phase calibration, and two antennas resolve only one path. Possible later with a ready-made coherent array (ESPARGOS). |
| Train in simulation (ray tracing), deploy on real boards | Not as a first step: room, body and hardware effects differ too much. Possible later as a simulation-to-real adaptation experiment. |
| Radio tomographic imaging | Kept as an optional training-free baseline for zones in week 4. |

## 10. Open questions
1. How to power and log three receivers at once.
2. Whether three links can resolve six zones about 0.75 m apart.
3. Whether the board-relative zone idea already exists in the literature.
4. A second person for data collection, needed before any paper.
5. Which two rooms to use.
