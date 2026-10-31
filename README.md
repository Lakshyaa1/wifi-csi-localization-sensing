# Wi-Fi CSI sensing with ESP32-S3

> **Work in progress.** The capture pipeline works on real hardware; a labelled dataset and
> accuracy results are not available yet.

Device-free human activity recognition from Wi-Fi **channel state information (CSI)**, using a
standard 2.4 GHz router as the transmitter and low-cost ESP32-S3 boards as receivers. Nothing is
worn or carried by the person.

The longer-term goal is to study **cross-room generalisation**: train in one room, measure the
accuracy drop in a second room, and recover part of it with source-free domain adaptation
(unlabelled data from the new room only), following the setting of
[MU-SHOT-Fi](https://arxiv.org/abs/2605.01369) (Radwan & Tabassum).

## How it works

```
router (2.4 GHz, fixed channel)  ── ping replies, 100/s ──►  ESP32-S3 receivers (×2)
                                                                  │ CSI over USB serial
                                                                  ▼
          record ─► quality check ─► clean (DSP) ─► dataset ─► classifier
```

- Each receiver pings the router 100 times a second and reports the CSI of every reply:
  52 data subcarriers of the 20 MHz legacy training field.
- Cleaning per clip: subcarrier reordering, gain-glitch removal, resampling to an exact 100 Hz
  grid, Hampel outlier filter, 20 Hz zero-phase low-pass.
- Activities: `empty`, `walk`, `sit`, `wave`, recorded as 5 s labelled clips in random order.

## Repository layout

| Folder | Contents |
|---|---|
| [`sensing/`](sensing/) | Current pipeline: `record.py` (with `cue.py`, an on-screen cue window), `check_quality.py`, `build_dataset.py`, `train_baseline.py`, `live.py` (live guess on screen), shared DSP in `csi_common.py`, and `sim.py` (simulated boards for testing without hardware) |
| [`firmware/`](firmware/) | Patch for Espressif's [esp-csi](https://github.com/espressif/esp-csi) `csi_recv_router` example: receiver gain lock and Wi-Fi power save off |
| [`docs/`](docs/) | Setup, data collection and pilot results |
| [`espnow_pilot/`](espnow_pilot/) | Scripts and one recording from the earlier ESP32-to-ESP32 (ESP-NOW) pilots |

## Documentation

- **[Data collection guide](docs/data_collection.md)**: flashing, placement, recording sessions,
  checking quality and training, plus known problems and fixes.
- **[Setup and pipeline reference](docs/setup.md)**: hardware, firmware settings, and each DSP step.
- **[Pilot runs](docs/pilot_runs.md)**: what has been tried so far and the results.

## Quick start

```bash
cd sensing
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib scikit-learn pyserial

# without hardware: simulated sessions, to check the pipeline runs
python sim.py --out data --room A && python sim.py --out data --room B
python build_dataset.py data/session_*_sim --out dataset.npz --plots figs
python train_baseline.py dataset.npz --source A --target B
```

With hardware, follow the [data collection guide](docs/data_collection.md).

## Status

| | |
|---|---|
| Motion detection, one ESP-NOW link (pilot 1) | 82% of 0.5 s windows, still vs motion, single recording ([details](docs/pilot_runs.md#pilot-1-motion-detection-one-esp-now-link-oct-2026)) |
| Four activities across sessions, ESP-NOW (pilot 2) | Chance level: too little data, poor placement ([details](docs/pilot_runs.md#pilot-2-four-activities-two-esp-now-receivers-oct-12-2026)) |
| Router-based capture (pilot 3) | Two receivers at ~100 packets/s; pipeline verified end to end on real data |
| Router-based activity session (pilot 4) | 80 clips recorded; walking not separable from an empty room with a router transmitter ([details](docs/pilot_runs.md#pilot-4-router-activity-session-oct-29-2026)) |
| Router activity sessions, small room (pilot 5) | 2 × 40 clips, 2 receivers, no lost clips. Learned on one session, tested on the other: **77.5%** both ways (chance 25%) ([details](docs/pilot_runs.md#pilot-5-small-room-two-sessions-oct-31-2026), [data](https://github.com/Lakshyaa1/wifi-csi-localization-sensing/releases/tag/data-2026-10-31)) |
| Labelled dataset, room A and room B | In progress: more sessions in room H, then a second room |
| Cross-room drop and source-free adaptation | Planned |

![Pilot 1: still vs motion](espnow_pilot/plots/first_motion_static_vs_motion.png)
