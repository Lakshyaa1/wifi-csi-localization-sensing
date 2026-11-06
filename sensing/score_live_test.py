"""
score_live_test.py - score the live classifiers on a scripted session of long blocks.

    python score_live_test.py <live test session folder> <dataset.npz>

Each block (e.g. 30 s of one action) is cut into 5 s windows every 0.5 s, the way live.py sees the
signal, and both live models guess each window:
  - lr:  logistic regression learned on every clip in dataset.npz (live.py --model lr)
  - cnn: CNN learned on sessions 1-3, adapted with SHOT-IM to session 4 without labels, 5 seeds averaged
         (live.py --model cnn)
Windows with a stall (gap over 0.25 s) or too few packets are skipped, as in recording.
Blocks with a label the models don't know (e.g. 'seated') are reported as a spread of guesses.
"""
import csv, sys, collections
import numpy as np
from csi_common import load_raw_log, process_clip
from build_dataset import DATA_SUBCARRIERS
from live import train, train_cnn_adapted
from train_baseline import features


def main():
    sdir, dataset = sys.argv[1], sys.argv[2]
    clf, labels, fs, clip_sec, _ = train(dataset)
    print("training the adapted CNN ...")
    cnn = train_cnn_adapted(dataset, [1, 2, 3], [4], 5)
    models = {"lr": lambda c: clf.predict_proba(features(c, fs)[None])[0], "cnn": cnn}
    boards = [load_raw_log(f"{sdir}/raw_board{i}.log") for i in (0, 1)]
    rows = list(csv.DictReader(open(f"{sdir}/index.csv", newline="")))
    n = int(clip_sec * fs)

    hits = {m: collections.Counter() for m in models}
    total = collections.Counter()
    confusion = {m: collections.defaultdict(collections.Counter) for m in models}
    for r in rows:
        label, b0, b1 = r["label"].strip(), float(r["t_start"]), float(r["t_end"])
        skipped = 0
        for t1 in np.arange(b0 + clip_sec, b1 + 1e-6, 0.5):
            arrs = []
            for b in boards:
                A, info = process_clip(b, t1 - clip_sec, t1, DATA_SUBCARRIERS, fs=fs, n=n)
                if A is None or not info["ok"]:
                    break
                arrs.append(A.T)
            if len(arrs) < len(boards):
                skipped += 1
                continue
            clip = np.stack(arrs)
            total[label] += 1
            for m, f in models.items():
                g = labels[int(np.argmax(f(clip)))]
                confusion[m][label][g] += 1
                hits[m][label] += g == label
        print(f"  block {r['clip_id']:>2} {label:7} {total[label]:3} windows so far, {skipped} skipped here")

    print(f"\nwindows per block label: {dict(total)}")
    for m in models:
        known = [l for l in total if l in labels]
        right = sum(hits[m][l] for l in known)
        count = sum(total[l] for l in known)
        print(f"\n{m}: {right}/{count} = {right / max(count, 1):.1%} of windows right (known actions only)")
        for l in sorted(total):
            spread = "  ".join(f"{g} {c / total[l]:.0%}" for g, c in confusion[m][l].most_common())
            print(f"  true {l:7} -> {spread}")


if __name__ == "__main__":
    main()
