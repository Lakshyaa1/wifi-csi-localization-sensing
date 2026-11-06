"""
train_cnn.py - a small CNN on the cleaned CSI, compared with the logistic regression baseline.

    python train_cnn.py dataset.npz

Input: each clip as an image, boards x subcarriers x time, as relative change per subcarrier
(the same scaling the baseline features use), reduced from 100 Hz to 50 Hz.
  1. Overfit check: can it memorise 16 clips? If not, the model or data handling is broken.
  2. Unseen sessions: test each session after learning on the other two, several seeds.
  3. Shuffled labels: the same test with wrong training labels should fall to about chance.
Settings are fixed in advance and scores are from the last epoch; test clips never choose anything.
"""
import argparse, time
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from train_baseline import features

EPOCHS, BATCH, LR, WEIGHT_DECAY, DROPOUT = 60, 16, 1e-3, 1e-4, 0.3
OVERFIT_STEPS = 200
POOL = "mean"     # final summary over time: "mean", or "meanmax" to also keep each pattern's peak


class SmallCNN(nn.Module):
    def __init__(self, boards, classes, dropout=DROPOUT, pool=None):
        super().__init__()

        def block(a, b, k):
            return nn.Sequential(nn.Conv2d(a, b, k, stride=2, padding=(k[0] // 2, k[1] // 2), bias=False),
                                 nn.BatchNorm2d(b), nn.ReLU())

        self.body = nn.Sequential(block(boards, 16, (3, 7)), block(16, 32, (3, 7)),
                                  block(32, 64, (3, 5)), block(64, 48, (3, 5)))
        self.pool = pool or POOL
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(96 if self.pool == "meanmax" else 48, classes))

    def forward(self, x):
        f = self.body(x).mean(dim=2)          # average over subcarriers: (clips, patterns, time)
        z = f.mean(dim=-1)
        if self.pool == "meanmax":
            z = torch.cat([z, f.max(dim=-1).values], dim=1)
        return self.head(z)


def prepare(X):
    """(clips, boards, subcarriers, time) -> relative change per subcarrier at 50 Hz."""
    X = X.astype(np.float32)
    X = X / (X.mean(axis=-1, keepdims=True) + 1e-6) - 1
    return torch.from_numpy(X[..., ::2].copy())


def fit(X, y, classes, seed, epochs=EPOCHS, batch=BATCH, dropout=DROPOUT, weight_decay=WEIGHT_DECAY):
    torch.manual_seed(seed)
    scale = X.std().item()
    model = SmallCNN(X.shape[1], classes, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(y))
        for i in range(0, len(y), batch):
            b = order[i:i + batch]
            if len(b) < 2:      # batch norm needs at least two clips
                continue
            opt.zero_grad()
            loss_fn(model(X[b] / scale), y[b]).backward()
            opt.step()
    return model, scale


@torch.no_grad()
def predict(model, scale, X):
    model.eval()
    return model(X / scale).argmax(1).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--train", type=int, nargs="+", help="session numbers to learn on, e.g. --train 1 3")
    ap.add_argument("--test", type=int, nargs="+", help="session numbers to test on; runs only this split")
    ap.add_argument("--pool", choices=["mean", "meanmax"], default="mean")
    a = ap.parse_args()
    global POOL
    POOL = a.pool
    d = np.load(a.dataset)
    X, y_np, session = prepare(d["X"]), d["y"], d["session"]
    y = torch.from_numpy(y_np.astype(np.int64))
    labels = [str(l) for l in d["labels"]]
    S = sorted(set(session))
    k = len(labels)
    n_weights = sum(p.numel() for p in SmallCNN(X.shape[1], k).parameters())
    print(f"{len(y)} clips, input {tuple(X.shape[1:])}, {n_weights} weights, non-finite values: "
          f"{int((~torch.isfinite(X)).sum())}, chance {1 / k:.0%}")
    print(f"final summary over time: {POOL}")
    print(f"settings: {EPOCHS} epochs, batch {BATCH}, lr {LR}, weight decay {WEIGHT_DECAY}, dropout {DROPOUT}")

    if a.test:
        chosen = lambda nums: np.isin(session, [S[i - 1] for i in nums])
        tr, te = np.where(chosen(a.train))[0], np.where(chosen(a.test))[0]
        print(f"\nLearn on sessions {a.train} ({len(tr)} clips), test on {a.test} ({len(te)} clips)")
        pooled, accs = np.zeros((k, k), int), []
        for seed in range(a.seeds):
            m, sc = fit(X[tr], y[tr], k, seed)
            p = predict(m, sc, X[te])
            accs.append((p == y_np[te]).mean())
            pooled += confusion_matrix(y_np[te], p, labels=range(k))
        print("  CNN per seed " + " ".join(f"{v:.1%}" for v in accs) + f", mean {np.mean(accs):.1%}")
        print(f"  CNN confusion summed over seeds, rows true, columns predicted {labels}")
        for l, row in zip(labels, pooled):
            print(f"  {l:6}", row.tolist(), f"right {row[labels.index(l)]}/{row.sum()}")
        F = np.stack([features(c, int(d['fs'])) for c in d["X"]])
        p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(F[tr], y_np[tr]).predict(F[te])
        print(f"  logistic regression {(p == y_np[te]).mean():.1%}, per class "
              + " ".join(f"{l} {(p[y_np[te] == i] == i).sum()}/{(y_np[te] == i).sum()}" for i, l in enumerate(labels)))
        return

    print("\n1. Overfit check: 4 clips per class from session 1, no dropout or weight decay")
    t0 = time.time()
    first = np.where(session == S[0])[0]
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        tr = np.concatenate([rng.choice(first[y_np[first] == c], 4, replace=False) for c in range(k)])
        m, s = fit(X[tr], y[tr], k, seed, epochs=OVERFIT_STEPS, batch=16, dropout=0.0, weight_decay=0.0)
        print(f"  seed {seed}: training clips right {(predict(m, s, X[tr]) == y_np[tr]).sum()}/16")
    print(f"  ({time.time() - t0:.0f} s)")

    print("\n2. Unseen sessions: test each session after learning on the other two")
    t0 = time.time()
    pooled, per_seed = np.zeros((k, k), int), []
    per_session = {s: [] for s in S}
    train_acc = []
    for seed in range(a.seeds):
        hits = 0
        for s in S:
            tr, te = np.where(session != s)[0], np.where(session == s)[0]
            m, sc = fit(X[tr], y[tr], k, seed)
            p = predict(m, sc, X[te])
            train_acc.append((predict(m, sc, X[tr]) == y_np[tr]).mean())
            per_session[s].append((p == y_np[te]).mean())
            pooled += confusion_matrix(y_np[te], p, labels=range(k))
            hits += (p == y_np[te]).sum()
        per_seed.append(hits / len(y))
        print(f"  seed {seed}: {hits}/{len(y)} = {hits / len(y):.1%}")
    for i, s in enumerate(S):
        print(f"  S{i + 1}: mean {np.mean(per_session[s]):.1%}, per seed "
              + " ".join(f"{v:.1%}" for v in per_session[s]))
    print(f"  all sessions: mean {np.mean(per_seed):.1%}, std across seeds {np.std(per_seed):.1%}")
    print(f"  training clips right at the end: mean {np.mean(train_acc):.1%}")
    print(f"  confusion summed over seeds, rows true, columns predicted {labels}")
    for l, row in zip(labels, pooled):
        print(f"  {l:6}", row.tolist())
    print(f"  ({time.time() - t0:.0f} s)")

    print("\n3. Shuffled training labels (should be near chance)")
    res = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(1000 + seed)
        hits = 0
        for s in S:
            tr, te = np.where(session != s)[0], np.where(session == s)[0]
            m, sc = fit(X[tr], y[tr][rng.permutation(len(tr))], k, seed)
            hits += (predict(m, sc, X[te]) == y_np[te]).sum()
        res.append(hits / len(y))
    print("  per seed " + " ".join(f"{v:.1%}" for v in res) + f", mean {np.mean(res):.1%}")


if __name__ == "__main__":
    main()
