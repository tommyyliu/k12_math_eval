"""How much does seeing *which* wrong answer a student picked help predict their other answers?

For each k, compare on held-out students after n random answers:
  binary        - MIRT, sees right/wrong
  nominal_rw    - nominal-response model, sees right/wrong only
  nominal_opt   - nominal-response model, sees the chosen option
The gap between the last two is the information in the choice of wrong answer.

Usage: uv run python scripts/run_nominal.py [--ks 1 3 6]
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from k12diag.data import correct_options, load_task34, make_split
from k12diag.evaluate import log_loss
from k12diag.mirt_cat import Posterior, item_probs, laplace_posterior
from k12diag.nominal import fit_nominal, laplace_nominal, observation_mask

ROOT = Path(__file__).resolve().parents[1] / "results"
OUT = ROOT / "nominal"
NS = [0, 1, 3, 5, 10, 20, 30, None]
N_SAMPLES = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 6])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = load_task34()
    split = make_split(df)
    tr, test = split.train, split.test
    correct = correct_options(df)
    rng = np.random.default_rng(0)
    orders = [rng.permutation(len(s.pool_items)) for s in test]  # same orders as run_dimensionality
    y_all = np.concatenate([s.target_y for s in test])

    rows = []
    for k in args.ks:
        t0 = time.time()
        path = OUT / f"nominal_k{k}.pkl"
        if path.exists():
            nom = pickle.loads(path.read_bytes())
        else:
            nom = fit_nominal(
                tr.student.to_numpy(), tr.item.to_numpy(), tr.AnswerValue.to_numpy() - 1, correct,
                split.n_students, split.n_items, k, verbose=True,
            )
            path.write_bytes(pickle.dumps(nom))
        mirt = pickle.loads((ROOT / "dimensionality" / f"mirt_k{k}.pkl").read_bytes())
        print(f"k={k} models ready ({time.time() - t0:.0f}s)")

        for n in NS:
            preds = {"binary": [], "nominal_rw": [], "nominal_opt": []}
            for s, order in zip(test, orders):
                o = order if n is None else order[:n]
                items, ys, opts = s.pool_items[o], s.pool_y[o].astype(float), s.pool_opt[o]

                mu, cov = laplace_posterior(mirt.A[items], mirt.d[items], ys)
                th = Posterior(mu, cov).samples(N_SAMPLES, rng)
                preds["binary"].append(item_probs(mirt, th)[:, s.target_items].mean(0))

                for name, see in (("nominal_rw", False), ("nominal_opt", True)):
                    M = observation_mask(nom, items, opts, see_option=see)
                    mu, cov = laplace_nominal(nom.B[items], nom.c[items], M)
                    th = Posterior(mu, cov).samples(N_SAMPLES, rng)
                    preds[name].append(nom.p_correct(th, s.target_items).mean(0))

            for name, ps in preds.items():
                per_student = [log_loss(p, s.target_y) for p, s in zip(ps, test)]
                p = np.concatenate(ps)
                rows.append({"k": k, "n": "all" if n is None else n, "variant": name,
                             "log_loss": float(np.mean(per_student)), "auc": roc_auc_score(y_all, p)})
            cur = [r for r in rows if r["k"] == k and r["n"] == ("all" if n is None else n)]
            print(f"k={k} n={'all' if n is None else n:>3}  " + "  ".join(
                f"{r['variant']}: LL {r['log_loss']:.4f} AUC {r['auc']:.4f}" for r in cur
            ) + f"  ({time.time() - t0:.0f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results.csv", index=False)
    cols = ["all" if n is None else n for n in NS]
    for metric in ("auc", "log_loss"):
        print(f"\n{metric}:")
        table = res.assign(n=res["n"].astype(str)).set_index(["k", "variant", "n"])[metric].unstack("n")
        print(table[[str(c) for c in cols]].round(4).to_string())


if __name__ == "__main__":
    main()
