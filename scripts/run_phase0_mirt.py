"""Phase 0 with multidimensional IRT: does adaptive selection reach the extra dimensions early?

Uses the MIRT fits from run_dimensionality.py.
Usage: uv run python scripts/run_phase0_mirt.py [--ks 1 3 6] [--n-test 1000]
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from k12diag.data import load_task34, make_split
from k12diag.mirt_cat import MDOptimal, MExpectedLossReduction, MRandom, full_information_mirt, run_student_mirt
from run_phase0 import summarize

ROOT = Path(__file__).resolve().parents[1] / "results"
MAX_Q = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 6])
    ap.add_argument("--n-test", type=int, default=1000)
    args = ap.parse_args()

    split = make_split(load_task34())
    test = split.test[: args.n_test]
    all_curves = []
    for k in args.ks:
        t0 = time.time()
        model = pickle.loads((ROOT / "dimensionality" / f"mirt_k{k}.pkl").read_bytes())
        out = ROOT / "phase0_mirt" / f"k{k}"
        out.mkdir(parents=True, exist_ok=True)
        policies = [MRandom(), MDOptimal(), MExpectedLossReduction(split.bank)]
        rng = np.random.default_rng(0)

        rows, trajs, preds = [], [], {}
        for s in test:
            full_ll, full_p = full_information_mirt(s, model, rng)
            preds.setdefault("full", []).append(full_p)
            for pol in policies:
                tr = run_student_mirt(s, pol, model, MAX_Q, rng)
                trajs.append({"student": tr.student, "policy": tr.policy, "items": tr.items,
                              "answers": tr.answers, "log_loss": tr.log_loss})
                for t, (ll, br) in enumerate(zip(tr.log_loss, tr.brier)):
                    rows.append((s.student, pol.name, t, ll, br, full_ll))
                for n in (0, 1, 3, 5, 10, 20, 30):
                    preds.setdefault((pol.name, n), []).append(tr.target_p[min(n, len(tr.target_p) - 1)])

        curves = pd.DataFrame(rows, columns=["student", "policy", "t", "log_loss", "brier", "full_info_log_loss"])
        curves.to_parquet(out / "curves.parquet")
        with open(out / "trajectories.pkl", "wb") as f:
            pickle.dump(trajs, f)

        y = np.concatenate([s.target_y for s in test])
        auc = {key: roc_auc_score(y, np.concatenate(v)) for key, v in preds.items()}
        print(f"\n===== k={k}  ({time.time() - t0:.0f}s) =====")
        print(f"full-information AUC {auc['full']:.4f}")
        for pol in policies:
            print(f"{pol.name:<8} AUC " + "  ".join(f"n={n}:{auc[(pol.name, n)]:.4f}" for n in (1, 3, 5, 10, 20, 30)))
        summarize(curves)
        all_curves.append(curves.assign(k=k))

    pd.concat(all_curves).to_parquet(ROOT / "phase0_mirt" / "curves_all.parquet")


if __name__ == "__main__":
    main()
