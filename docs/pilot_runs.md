# Pilot runs

Three small pilot runs so far. None is a full experiment; together they shaped the current setup.

## Pilot 1: motion detection, one ESP-NOW link (Oct 2026)

- **Hardware:** two ESP32-S3 boards about 80 cm apart, line of sight. Espressif's esp-csi
  `csi_send` / `csi_recv` examples, ESP-NOW on channel 11, about 70 CSI packets/s.
- **Recording:** one 50 s session with spoken cues: still, motion, still, motion, still (10 s
  each). Motion was a hand waving through the link.
- **Method:** amplitude of the 52 active L-LTF subcarriers; motion score = mean per-subcarrier
  standard deviation over 0.5 s windows; a single threshold.

| | Still | Motion |
|---|---|---|
| Median motion score | 0.95 | 1.58 (1.7×) |

- The threshold classifies **82% of 0.5 s windows** correctly (93% of still windows, 70% of motion
  windows). The threshold was chosen on the same recording, so this is not a held-out result.
- Most missed motion windows fall in the first ~3 s of each motion phase, while moving into
  position after the cue.
- At this short range, motion mostly shadows the direct path: all subcarriers dim together,
  so RSSI reacts too.

![Static vs motion](../espnow_pilot/plots/first_motion_static_vs_motion.png)

![CSI amplitude](../espnow_pilot/plots/first_motion.png)

After removing the linear phase trend across subcarriers, the phase is stable when still and
fluctuates about twice as much during motion:

![Phase](../espnow_pilot/plots/first_motion_phase.png)

## Pilot 2: four activities, two ESP-NOW receivers (Oct 12, 2026)

- **Hardware:** one transmitter, two receivers (receivers 1.33 m apart), packets joined across
  receivers by the transmitter's packet counter.
- **Recording:** two short sessions, `empty`, `still`, `walk`, `jump`, at one spot.
- **Method:** hand-made features per 2 s window; Random Forest and RBF-SVM; trained on one
  session and tested on the other.

| Train → test | Majority class | Random Forest | SVM | Shuffled labels (RF) |
|---|---|---|---|---|
| session 1 → 2 | 25.0% | 23.3% | 26.7% | 11.2% |
| session 2 → 1 | 25.0% | 28.4% | 32.8% | 35.3% |

**Result: chance level across sessions.** Likely causes: the actor stood outside the first
Fresnel zone of the links (about 0.25 m wide at that distance), slow drift between sessions,
and very little data.

## Pilot 3: router as transmitter (Oct 24, 2026)

- **Hardware:** a 2.4 GHz router (channel 1, 20 MHz) and two ESP32-S3 receivers running
  `csi_recv_router` with the [firmware patch](../firmware/csi_recv_router.patch); 52 subcarriers.
- **What was established:**
  - Both receivers stream at a steady ~100 packets/s together, once Wi-Fi power save is
    turned off (without it the rate collapses after a few seconds).
  - The subcarrier order of the router firmware's L-LTF buffer was checked on real data.
  - Capture, quality check and dataset building run end to end on real data.
- **Not yet established:** activity accuracy. Only 11 usable clips were recorded, far too few
  to train on. Even in an empty room the amplitude changed about 28% between consecutive
  packets, which is being investigated (see the
  [data collection guide](data_collection.md#8-problems-we-already-hit)).

## Pilot code

The ESP-NOW pilots used the scripts in [`espnow_pilot/`](../espnow_pilot/):

| File | Purpose |
|---|---|
| `logger.py` | Record labelled CSI from one or more `csi_recv` receivers (`boards.csv` maps USB serials to roles) |
| `merge.py` | Merge recordings made on two laptops, by packet id |
| `preprocess.py` | Join receivers on packet id, filter, resample, window |
| `features.py`, `train_classical.py` | Hand-made features; Random Forest / SVM with majority and shuffled-label checks |
| `plot_csi.py`, `plot_session.py`, `static_vs_motion.py`, `analyze_csi.py`, `phase_check.py` | Plots and pilot 1 analysis |
| `live_view.py` | Live activity level and amplitude view from connected receivers |
| `layout.py` | Board and zone coordinates |

Known issue: `preprocess.py` reorders subcarriers as if the ESP-NOW buffer were 0..31, −32..−1.
On our boards that buffer is already in frequency order, so the reorder (and the phase
sanitization that depends on it) is wrong. Amplitude features per subcarrier are unaffected.
