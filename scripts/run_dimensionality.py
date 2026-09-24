"""How many latent dimensions does Eedi performance support?

For k = 1..K, fit MIRT on training students, then for each held-out student observe n random
answers from their pool (n = 0, 1, 3, ..., all) and score predictions on their target set.

Usage: uv run python scripts/run_dimensionality.py
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from k12diag.data import load_task34, make_split
from k12diag.evaluate import log_loss
from k12diag.mirt import fit_mirt, infer_students, predict

OUT = Path(__file__).resolve().parents[1] / "results" / "dimensionality"
KS = [1, 2, 3, 4, 6, 8, 10]
NS = [0, 1, 3, 5, 10, 20, 30, None]  # None = entire pool


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    split = make_split(load_task34())
    tr = split.train
    test = split.test
    rng = np.random.default_rng(0)
    orders = [rng.permutation(len(s.pool_items)) for s in test]  # same random order for every k

    tgt_s = np.concatenate([np.full(len(s.target_items), i) for i, s in enumerate(test)])
    tgt_j = np.concatenate([s.target_items for s in test])
    tgt_y = np.concatenate([s.target_y for s in test])

    rows = []
    for k in KS:
        path = OUT / f"mirt_k{k}.pkl"
        if path.exists():
            model = pickle.loads(path.read_bytes())
        else:
            model = fit_mirt(
                tr.student.to_numpy(), tr.item.to_numpy(), tr.IsCorrect.to_numpy().astype(np.float32),
                split.n_students, split.n_items, k, verbose=True,
            )
            path.write_bytes(pickle.dumps(model))
        for n in NS:
            obs = [o if n is None else o[:n] for o in orders]
            s_ = np.concatenate([np.full(len(o), i) for i, o in enumerate(obs)])
            j_ = np.concatenate([test[i].pool_items[o] for i, o in enumerate(obs)])
            y_ = np.concatenate([test[i].pool_y[o] for i, o in enumerate(obs)]).astype(np.float32)
            mu, sigma = infer_students(model, s_, j_, y_, len(test))
            p = predict(model, mu, sigma, tgt_s, tgt_j)
            per_student = pd.Series(
                -(tgt_y * np.log(np.clip(p, 1e-9, 1)) + (1 - tgt_y) * np.log(np.clip(1 - p, 1e-9, 1)))
            ).groupby(tgt_s).mean()
            rows.append({
                "k": k, "n": "all" if n is None else n,
                "log_loss": per_student.mean(),  # mean over students, as in phase 0
                "log_loss_rows": log_loss(p, tgt_y),
                "auc": roc_auc_score(tgt_y, p),
                "acc": float(np.mean((p > 0.5) == tgt_y)),
            })
            r = rows[-1]
            print(f"k={k:2d} n={str(r['n']):>3}  LL {r['log_loss']:.4f}  AUC {r['auc']:.4f}  acc {r['acc']:.4f}"
                  f"   ({time.time() - t0:.0f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results.csv", index=False)
    cols = ["all" if n is None else n for n in NS]
    for metric in ("auc", "log_loss"):
        print(f"\n{metric} by k (rows) and answers observed (columns):")
        print(res.pivot(index="k", columns="n", values=metric)[cols].round(4).to_string())


if __name__ == "__main__":
    main()
