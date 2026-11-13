"""
wimans_holdout.py - Part A of wimans_transfer.py again, with adaptation and scoring on separate clips.

    python wimans_holdout.py <wimans root>

Learn in the classroom (4 classes: empty, sit_down, walk, wave; 2.4 GHz; 0-1 person), then in each other room:
adapt on a random half of its clips (balanced by class; labels only pick the split), score the other half,
swap, as adapt.py --holdout. Inputs: all 9 antenna pairs, and cut to 2 pairs. 5 seeds.
"""
import sys
import numpy as np
import torch
from wimans_transfer import load_wimans, CUT, SEEDS
from train_cnn import fit, prepare
from adapt import adabn, shot_im, predict_proba


def main():
    torch.set_num_threads(16)
    X, act, env = load_wimans(sys.argv[1])
    classes = ["empty", "sit_down", "walk", "wave"]
    keep = np.isin(act, classes)
    y_np = np.array([classes.index(a) for a in act[keep]])
    e = env[keep]
    y = torch.from_numpy(y_np.astype(np.int64))
    for name, Xi in {"9 pairs": X[keep].reshape(keep.sum(), 9, 52, 300),
                     "2 pairs (cut)": np.stack([X[keep][:, t, r] for t, r in CUT], 1)}.items():
        P = prepare(Xi)
        src = np.where(e == "classroom")[0]
        print(f"\n[{name}]")
        for room in ("meeting_room", "empty_room"):
            tgt = np.where(e == room)[0]
            yt = y_np[tgt]
            acc = {"source only": [], "AdaBN": [], "SHOT-IM": []}
            for seed in range(SEEDS):
                m, s = fit(P[src], y[src], 4, seed)
                rng = np.random.default_rng(100 + seed)
                half = np.zeros(len(tgt), bool)
                for c in range(4):
                    idx = rng.permutation(np.where(yt == c)[0])
                    half[idx[:len(idx) // 2]] = True
                preds = {k_: np.zeros(len(tgt), int) for k_ in acc}
                for a_on, s_on in ((half, ~half), (~half, half)):
                    Xa, Xs = P[tgt[a_on]], P[tgt[s_on]]
                    for k_, mm in (("source only", m), ("AdaBN", adabn(m, s, Xa)), ("SHOT-IM", shot_im(m, s, Xa, seed))):
                        preds[k_][s_on] = predict_proba(mm, s, Xs).argmax(1).numpy()
                for k_ in acc:
                    acc[k_].append((preds[k_] == yt).mean())
            print(f"  classroom -> {room:12} " + "   ".join(
                f"{k_} {np.mean(v):.1%} ({min(v):.1%}-{max(v):.1%})" for k_, v in acc.items()), flush=True)


if __name__ == "__main__":
    main()
