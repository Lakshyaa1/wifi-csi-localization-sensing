"""
build_dataset.py - turn recorded sessions into a training-ready dataset, and make the DSP plots.

    python build_dataset.py data/session_A1 data/session_A2 data/session_B1 --out dataset.npz --plots figs

For every clip in each session's index.csv, and for every board:
  1. CSI ints -> complex -> amplitude, subcarriers in frequency order
  2. keep the 52 data subcarriers (drop DC and guard bands)
  3. drop packets where every subcarrier jumps together (gain glitches)
  4. resample onto an exact 100 Hz grid (packets arrive with jitter and some are lost)
  5. Hampel filter: replace remaining single-subcarrier spikes with the local median
  6. low-pass at 20 Hz (zero-phase Butterworth)
Boards are stacked, giving one array per clip of shape (boards, 52, 300) (300 for a 3 s clip at 100 Hz).
That is the same idea as the paper's 1 x F x T input, with F = boards x subcarriers.
"""
import argparse, csv, glob, json, os
import numpy as np
from csi_common import (load_raw_log, to_complex, resample, process_clip, motion_component,
                        spectrogram)

FS = 100
DATA_SUBCARRIERS = np.r_[6:32, 33:59]      # k = -26..-1 and 1..26 in the 64-point L-LTF layout


def load_session(sdir):
    meta = json.load(open(os.path.join(sdir, "meta.json")))
    with open(os.path.join(sdir, "index.csv")) as f:
        clips = list(csv.DictReader(f))
    boards = {}
    for p in sorted(glob.glob(os.path.join(sdir, "raw_board*.log"))):
        boards[os.path.basename(p)[4:-4]] = load_raw_log(p)
    return meta, clips, boards


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--out", default="dataset.npz")
    ap.add_argument("--clip-sec", type=float, default=None, help="default: from the first session's meta.json")
    ap.add_argument("--keep-flagged", action="store_true", help="also keep clips flagged low_rate or gap")
    ap.add_argument("--plots", default=None, help="folder for DSP figures (one example per class)")
    a = ap.parse_args()
    if a.clip_sec is None:
        a.clip_sec = float(json.load(open(os.path.join(a.sessions[0], "meta.json"))).get("clip_sec", 3.0))
    T = int(a.clip_sec * FS)
    print(f"clip length {a.clip_sec:g} s -> {T} samples")

    X, y_names, rooms, sess, cids, persons, qinfo = [], [], [], [], [], [], []
    examples = {}
    n_board_ref = None
    for sdir in a.sessions:
        meta, clips, boards = load_session(sdir)
        names = sorted(boards)
        if n_board_ref is None:
            n_board_ref = len(names)
        if len(names) != n_board_ref:
            print(f"skip {sdir}: {len(names)} boards, expected {n_board_ref}")
            continue
        ncsi = boards[names[0]]["csi"].shape[1]
        sub = DATA_SUBCARRIERS if ncsi == 128 else np.arange(ncsi // 2)
        kept = 0
        for c in clips:
            if c["flag"] != "ok" and not a.keep_flagged:
                continue
            t0 = float(c["t_start"]); t1 = t0 + a.clip_sec
            arrs, infos = [], []
            for nm in names:
                A, info = process_clip(boards[nm], t0, t1, sub, fs=FS, n=T)
                arrs.append(A); infos.append(info)
            if any(A is None for A in arrs):
                continue
            X.append(np.stack([A.T for A in arrs]).astype(np.float32))      # (boards, S, T)
            y_names.append(c["label"]); rooms.append(c["room"]); sess.append(os.path.basename(sdir))
            cids.append(int(c["clip_id"])); persons.append(c.get("person", ""))
            qinfo.append(json.dumps(infos))
            kept += 1
            if c["label"] not in examples:
                examples[c["label"]] = (boards[names[0]], t0, t1, sub, arrs[0])
        print(f"{os.path.basename(sdir)}: room {meta['room']}, {kept}/{len(clips)} clips kept, "
              f"{len(names)} boards, {len(sub)} subcarriers")

    if not X:
        raise SystemExit("no clips kept")
    labels = sorted(set(y_names))
    y = np.array([labels.index(n) for n in y_names])
    X = np.stack(X)
    np.savez_compressed(a.out, X=X, y=y, labels=np.array(labels), room=np.array(rooms),
                        session=np.array(sess), clip_id=np.array(cids), person=np.array(persons),
                        quality=np.array(qinfo), fs=FS)
    print(f"\nsaved {a.out}:  X {X.shape} (clips, boards, subcarriers, time),  classes {labels}")
    for r in sorted(set(rooms)):
        cnt = {lab: int(((y == i) & (np.array(rooms) == r)).sum()) for i, lab in enumerate(labels)}
        print(f"  room {r}: {cnt}")

    if a.plots:
        make_plots(examples, a.plots, T)


def make_plots(examples, outdir, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    labs = sorted(examples)

    # 1. raw vs cleaned for one clip (the DSP story)
    lab = next((l for l in labs if l != "empty"), labs[0])
    board, t0, t1, sub, clean = examples[lab]
    t = board["board_t"]; sel = (t >= t0 - 0.5) & (t <= t1 + 0.5)
    amp = np.abs(to_complex(board["csi"][sel], board["first_word_invalid"][sel]))[:, sub]
    raw_t = t[sel] - t0
    _, uni = resample(t[sel], amp, t0, T, FS)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    j = int(np.argmax(clean.std(axis=0)))
    ax[0].plot(raw_t, amp[:, j], ".", ms=3, label="received packets")
    ax[0].plot(np.arange(T) / FS, clean[:, j], lw=2, label="cleaned (glitches, resample, Hampel, low-pass)")
    ax[0].set(title=f"'{lab}': one subcarrier, raw vs cleaned", xlabel="time (s)", ylabel="amplitude", xlim=(0, T / FS))
    ax[0].legend(fontsize=8)
    for k, (M, ttl) in enumerate([(uni, "resampled only"), (clean, "fully cleaned")]):
        ax[k + 1].imshow(M.T, aspect="auto", origin="lower", extent=[0, T / FS, 0, M.shape[1]], cmap="viridis")
        ax[k + 1].set(title=f"all subcarriers, {ttl}", xlabel="time (s)", ylabel="subcarrier")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "dsp_raw_vs_clean.png"), dpi=110); plt.close(fig)

    # 2. one example per class: heatmap, motion component, spectrogram
    fig, ax = plt.subplots(len(labs), 3, figsize=(15, 2.8 * len(labs)), squeeze=False)
    for r, lab in enumerate(labs):
        A = examples[lab][4]
        m = motion_component(A)
        f, ts, S = spectrogram(m, FS)
        ax[r, 0].imshow(A.T, aspect="auto", origin="lower", extent=[0, T / FS, 0, A.shape[1]], cmap="viridis")
        ax[r, 0].set(ylabel=lab, title="amplitude (board 0)" if r == 0 else "")
        ax[r, 1].plot(np.arange(T) / FS, m, lw=1)
        ax[r, 1].set(title="strongest motion component (PCA 1)" if r == 0 else "")
        ax[r, 2].pcolormesh(ts, f, 20 * np.log10(S + 1e-6), shading="auto", cmap="magma")
        ax[r, 2].set(ylim=(0, 30), title="spectrogram of PCA 1 (Hz)" if r == 0 else "")
    for c in range(3):
        ax[-1, c].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "classes.png"), dpi=110); plt.close(fig)
    print(f"saved figures in {outdir}/")


if __name__ == "__main__":
    main()
