"""Combine recordings of the same session made on different laptops.

Usage:
  python merge.py s01_roomA s01_roomA_rx1 [more ...]      # writes data/s01_roomA_merged.csv
The first recording is the labelled one (made with --preset, voice cues). The others
can be recorded with one long phase, e.g. on the second laptop:
  python logger.py --links rx1=/dev/ttyACM0 --phases raw:600 --name s01_roomA_rx1
started before and stopped (Ctrl+C) after the labelled recording. Their rows get the
activity and zone of the labelled recording by packet id (same id = same transmission),
so the laptops' clocks do not need to agree. Packets outside the labelled recording's
id range are dropped. The transmitter must stay powered throughout (ids restart on boot).
"""
import csv
import sys
from bisect import bisect_right
from pathlib import Path

DATA = Path(__file__).parent / "data"


def main():
    main_name, others = sys.argv[1], sys.argv[2:]
    with open(DATA / f"{main_name}.csv") as f:
        reader = csv.DictReader(f)
        cols, rows = reader.fieldnames, list(reader)
    ids = [int(r["id"]) for r in rows]
    if ids != sorted(ids):
        sys.exit("packet ids go backwards in the labelled recording: was the transmitter restarted?")
    main_links = {r["link"] for r in rows}
    out, counts = list(rows), {}
    for name in others:
        with open(DATA / f"{name}.csv") as f:
            for r in csv.DictReader(f):
                i = int(r["id"])
                if r["link"] in main_links:
                    sys.exit(f"{name} has link {r['link']}, which is already in {main_name}")
                if not ids[0] <= i <= ids[-1]:
                    continue
                ref = rows[bisect_right(ids, i) - 1]      # labelled packet at or just before this id
                r["activity"], r["zone"] = ref["activity"], ref["zone"]
                out.append(r)
                counts[r["link"]] = counts.get(r["link"], 0) + 1
    path = DATA / f"{main_name}_merged.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f"{main_name}: {len(rows)} labelled packets on {sorted(main_links)}; added "
          + ", ".join(f"{k} {v}" for k, v in counts.items()) + f" -> {path}")


if __name__ == "__main__":
    main()
