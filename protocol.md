# Recording protocol

One **session** = one ~8 min recording of one person doing every activity at every zone, made with:

```bash
cd ~/projects/wifi-csi-localization-sensing
.venv/bin/python logger.py --list                       # all receivers show a role rx1..rx3
.venv/bin/python logger.py --preset joint --name s01_roomA --speak --delay 20
.venv/bin/python preprocess.py s01_roomA
```

Name sessions `sNN_<room>` (`s01_roomA` … `s04_roomA`, then `s05_roomB` …).

- **Room A (source):** 4 sessions over at least 2 different days. Sessions 1–3 train, session 4 tests.
- **Room B (target):** 3 sessions. Sessions 1–2 are used without labels for adaptation, session 3 tests.

## Layout (identical in both rooms)

Four boards at the corners of a **3 m × 3 m square**, all about 1 m high, antennas facing the centre. Positions in metres (also in `layout.py`):

| Board | (x, y) | Connection |
|---|---|---|
| tx | (0, 0) | charger or power bank |
| rx1 | (3, 0) | USB to laptop |
| rx2 | (3, 3) | USB to laptop |
| rx3 | (0, 3) | USB to laptop |

Six zones, each a taped spot on the floor at the zone centre:

| Zone | (x, y) | Zone | (x, y) |
|---|---|---|---|
| 1 | (0.75, 1.0) | 4 | (0.75, 2.0) |
| 2 | (1.5, 1.0) | 5 | (1.5, 2.0) |
| 3 | (2.25, 1.0) | 6 | (2.25, 2.0) |

- Tape around every board (position and orientation) and photograph the setup once per room.
- Each board keeps its role and corner in every session and in both rooms. `boards.csv` maps each board's USB serial number to its role; label the boards to match.
- Always plug a board into the same USB socket on the board (the serial number differs between its two sockets).

## Session sequence (`--preset joint`)

A 5 s transition with the cue "next, …" comes before each block: use it to move and start the activity, so you are already doing it at "go". Transition packets are dropped automatically.

| Block | Duration | What to do |
|---|---|---|
| empty | 30 s | Stand outside the square, at least 2 m from any link |
| still | 20 s per zone | Stand on the spot, arms by your side, facing the +y side |
| walk | 20 s per zone | Walk on the spot, or in a small circle that stays inside the zone |
| jump | 20 s per zone | Jump on the spot, about once a second |

Order: empty, then zone 1 (still, walk, jump), zone 2, … zone 6. `--preset short` is a 1 min version (empty plus the three activities at zone 1) for checking the setup.

## Notes to write per session (`data/sessions.csv`)

`session, date, room, people_in_room, door_open, notes`

## Rules

- Do not move or touch the boards during or between sessions in the same room.
- Only one person in the room during a session.
- The transmitter stays powered for the whole session; the logger waits for steady packets on every link before it starts.
- If something goes wrong in a phase, finish the session and write it in `notes`; do not cut the CSV by hand.
