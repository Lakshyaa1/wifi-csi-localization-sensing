# Plan: adaptable joint sensing and localization with ESP32-S3 CSI

Written 2026-10-11, status updated 2026-10-12. Covers Oct 10 to about Nov 7.

## The goal in one sentence

One model that tells **what a person is doing and where they are**, using only Wi-Fi CSI between ESP32-S3 boards (nothing carried by the person), and that **adapts to a new room without any labels from that room**.

## The idea that makes it work

Position fingerprints normally belong to one room: "spot A in room 1" has no meaning in room 2, so there is nothing to adapt. Here the zones are defined **relative to the boards**, not the room:

- 4 boards at the corners of a 3 m × 3 m square, the same in every room.
- 6 zones inside that square, numbered the same way in every room.

"Zone 4" then means the same thing anywhere the square is set up. That lets a label-free adaptation method (the kind used in MU-SHOT-Fi for activities) be applied to the **position** output as well as the activity output.

What we can honestly claim if it works: a real ESP32 dataset with activity and position labelled together in two rooms, a joint model, and source-free adaptation of both outputs. What we cannot claim: a new adaptation algorithm. **Whether the board-frame idea is new still needs a literature check** (week 1 task).

## The flow

```
record (4 boards)  ->  preprocess  ->  joint model  ->  test in same room
                                            |
                                            v
                     new room, no labels -> adapt -> test in new room
```

| Step | Code | Status |
|---|---|---|
| Record labelled CSI from several receivers | `logger.py`, `boards.csv`, `protocol.md` | works on one real link (about 70 packets/s); several receivers at once not yet tried |
| Join receivers on packet id, filter, window | `preprocess.py` | works on one real link; the multi-receiver join is tested on synthetic data only |
| Classical baseline (Random Forest, SVM) | `features.py`, `train_classical.py` | written, tested on synthetic data |
| Joint CNN with activity and zone outputs | `dataset.py`, `models.py`, `train.py` | week 2 |
| Adaptation to room B | `adapt.py` | week 3 |

## Hardware status (Oct 12)
- Three boards identified and flashed: one transmitter, two receivers (`boards.csv`). The fourth has not been seen yet.
- The laptop has only handled one board at a time so far. Three receivers need a powered USB hub, or a second laptop for logging.
- No labelled session has been recorded yet.

## Week by week

### Week 1 (Oct 10–17): get real data flowing
- Two-board test of the new logger as soon as the second cable arrives.
- Flash and register receivers 2 and 3; get three 3 m USB data cables.
- Tape the 3 m square and six zones in room A; record sessions 1–2 (8 min each).
- Run the classical baseline on them.
- Literature check on board-relative labels for device-free localization.
- **Learn:** softmax and cross-entropy; entropy and diversity (lessons 1–2, done); what a Random Forest does.

### Week 2 (Oct 18–24): joint model in one room
- Record room A sessions 3–4 on other days.
- Build the CNN with two output heads and train it.
- Compare on the held-out session: Random Forest vs CNN, 1 receiver vs 3, separate vs joint training.
- **Learn:** convolution, BatchNorm, training loop, overfitting, why the split is by session.

### Week 3 (Oct 25–31): second room and adaptation
- Set up the same square in room B; record 3 sessions.
- Measure the drop of the room A model in room B.
- Adaptation ladder, 3 seeds each: no adaptation, AdaBN, information maximisation, plus a self-supervised task.
- **Learn:** domain shift, AdaBN, information maximisation and collapse, self-supervision.

### Week 4 (Nov 1–7): ablations and write-up (cut first if time is short)
- Does removing the static background make features more room-independent?
- Optional tomography baseline for zones (no training needed).
- README with setup photo, result tables and limitations.

## Decision points
- **End of week 2:** do 3 receivers beat 1 for zones? If not, rethink the layout before recording room B.
- **End of week 3:** is there a clear drop in room B and a recovery that holds over 3 seeds? If yes, a workshop or work-in-progress paper is worth writing, after adding a second person. If no, this stays a public repo and a learning project.

## If the hardware is delayed
- Record with two boards (one link) to practise the pipeline; that data does not go into the final 4-board set.
- Build the joint model and adaptation code on WiMANS, which is already downloaded and has activity and location labels across rooms.

## Rules
- Split train and test by session, never by window.
- Room B labels are used only for the final evaluation, never for adaptation or model selection.
- Report last-epoch results over 3 seeds.
- Nothing is pushed until a phase has a real result.

## Risks
- One person and two rooms is a small study; a second person is needed before any paper.
- Six zones of about 0.75 m may be finer than three links can resolve.
