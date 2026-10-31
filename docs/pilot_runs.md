# Pilot runs

Four small pilot runs so far. None is a full experiment; together they shaped the current setup.

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

## Pilot 4: router activity session (Oct 29, 2026)

- **Hardware:** a shared 2.4 GHz router (channel 11) as transmitter, one ESP32-S3 receiver at
  about 3 m (RSSI about −40 dBm). A second receiver averaged only 13 packets/s and was dropped.
- **Recording:** 80 clips of 3 s (20 each of `empty`, `walk`, `sit`, `wave`), shuffled, with
  voice and on-screen cues. The laptop pinged the router at 100 Hz during the session, which
  kept the reply rate up; without it the rate fell to about 10/s.
- **Packets per clip (median, ideal 300):** empty 266, walk 216, sit 54, wave 38. Packets were
  lost when the actor stood on the link, so 30 of 80 clips were too sparse to use.
- **Motion score** (mean over subcarriers of the relative amplitude change in a clip), clips
  with at least 150 packets:

| | empty (13) | walk (17) |
|---|---|---|
| all frequencies | 0.251 | 0.264 |
| 0.3–5 Hz band | 0.162 | 0.202 |

- **Result: walking was not separable from the empty room.** Normalising each packet by its mean
  level did not change this, so it is not simple gain jitter.
- **Unconfirmed suspects:** the router spreading frames over several antennas (pilot 1's
  single-antenna ESP transmitter was much steadier), and an "empty" room that wasn't empty
  (the actor stayed in the room). Next: ESP-to-ESP link, actor outside the room for `empty`,
  and a 1-minute empty-vs-walk check before any full session.

## Pilot 5: small room, two sessions (Oct 31, 2026)

- **Hardware:** a 2.4 GHz router in the same small room (channel 10, RSSI about −35 to −50 dBm)
  as transmitter, two ESP32-S3 receivers. Both boards set to ignore routers weaker than −70 dBm.
- **1-minute checks first** (30–45 s each of empty, wave, walk, sit at the final layout): every
  action scored above the empty room, so the full sessions went ahead.
- **Recording:** two sessions of 40 clips of 5 s (10 each of `empty`, `walk`, `sit`, `wave`),
  shuffled, red/green screen, laptop ping on. All 80 clips flagged `ok`: about 500 packets per
  clip on both boards, longest gap 0.19 s. `empty` = actor outside the room.
- **Motion score** (mean over subcarriers of std / mean in a clip), median per class:

| | empty | sit | wave | walk |
|---|---|---|---|---|
| session 1, board 0 | 0.088 | 0.189 | 0.195 | 0.322 |
| session 1, board 1 | 0.048 | 0.112 | 0.122 | 0.227 |
| session 2, board 0 | 0.106 | 0.170 | 0.143 | 0.290 |
| session 2, board 1 | 0.095 | 0.101 | 0.127 | 0.219 |

- **Sit vs wave** have similar scores but differ in speed: sit is one slow change (share of
  energy above 1 Hz about 0.08), wave repeats about once a second (about 0.3–0.45).
- **Baseline across sessions** (`train_baseline.py` features, 16 per clip; logistic regression;
  trained on one whole session, tested on the other):

| learned on → tested on | accuracy | |
|---|---|---|
| session 1 → session 2 | 77.5% (31/40) | walk 10/10, sit 9/10, wave 9/10, empty 3/10 |
| session 2 → session 1 | 77.5% (31/40) | walk 10/10, empty 8/10, wave 7/10, sit 6/10 |
| shuffled labels (20 tries) | 22.5% | chance is 25% |

- With 40 test clips the 95% range is about 62–88%. Same room, same day, same person.
- **Main error: empty vs sit.** Session 2's empty room was noisier (cause unknown), and a quiet,
  slowly drifting signal looks like a sit.
- **Live test:** `live.py` guesses from the last 5 s every 0.5 s. It gives the same answer as the
  offline pipeline on all 80 recorded clips. In a first 2-minute live try, it rarely said `empty`.
- Data: [release data-2026-10-31](https://github.com/Lakshyaa1/wifi-csi-localization-sensing/releases/tag/data-2026-10-31).

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
