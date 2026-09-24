"""Phase 0, first pass: 2PL IRT + grid posterior, comparing selection policies on held-out Eedi students.

Usage: uv run python scripts/run_phase0.py
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from k12diag.data import load_task34, make_split
from k12diag.evaluate import full_information, run_student
from k12diag.irt import fit_2pl
from k12diag.policies import DifficultyPolicy, ExpectedLossReductionPolicy, FisherPolicy, RandomPolicy

OUT = Path(__file__).resolve().parents[1] / "results" / "phase0_2pl"
MAX_Q = 30


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = load_task34()
    split = make_split(df)
    tr = split.train
    print(f"train rows {len(tr):,}  students {tr.student.nunique():,}  test students {len(split.test)}")
    print(f"bank {len(split.bank)} of {split.n_items} items (>= 100 training responses)")
    print(f"pool size median {np.median([len(s.pool_items) for s in split.test]):.0f}, "
          f"target size median {np.median([len(s.target_items) for s in split.test]):.0f}")

    model_path = OUT / "model.pkl"
    if model_path.exists():
        model = pickle.loads(model_path.read_bytes())
    else:
        model = fit_2pl(
            tr.student.to_numpy(), tr.item.to_numpy(), tr.IsCorrect.to_numpy().astype(float),
            split.n_students, split.n_items, verbose=True,
        )
        model_path.write_bytes(pickle.dumps(model))
    print(f"model ready ({time.time() - t0:.0f}s)")

    P = model.p_correct_grid()
    policies = [
        RandomPolicy(), DifficultyPolicy(), FisherPolicy(model), ExpectedLossReductionPolicy(split.bank),
    ]
    rng = np.random.default_rng(0)

    rows, trajectories = [], []
    for s in split.test:
        full_ll, _ = full_information(s, P, model.log_prior)
        for pol in policies:
            tr_ = run_student(s, pol, P, model.log_prior, MAX_Q, rng)
            trajectories.append(tr_)
            for t, (ll, br) in enumerate(zip(tr_.log_loss, tr_.brier)):
                rows.append((s.student, pol.name, t, ll, br, full_ll))
    print(f"evaluation done ({time.time() - t0:.0f}s)")

    curves = pd.DataFrame(rows, columns=["student", "policy", "t", "log_loss", "brier", "full_info_log_loss"])
    curves.to_parquet(OUT / "curves.parquet")
    with open(OUT / "trajectories.pkl", "wb") as f:
        pickle.dump(
            [{k: v for k, v in vars(t).items() if k != "target_p"} for t in trajectories], f
        )
    with open(OUT / "split.pkl", "wb") as f:
        pickle.dump({"test": split.test, "bank": split.bank}, f)

    summarize(curves)


def questions_to_fraction(mean_curve: np.ndarray, zero: float, full: float, frac: float) -> float:
    gain = (zero - mean_curve) / (zero - full)
    hit = np.nonzero(gain >= frac)[0]
    return float(hit[0]) if len(hit) else np.nan


def summarize(curves: pd.DataFrame, n_boot: int = 500):
    rng = np.random.default_rng(0)
    students = curves.student.unique()
    wide = curves.pivot_table(index=["policy", "student"], columns="t", values="log_loss")
    full = curves.groupby("student").full_info_log_loss.first()

    print(f"\nzero-question log loss {wide.xs('random')[0].mean():.4f}   full-information {full.mean():.4f}")
    print(f"{'policy':<12}{'LL@5':>8}{'LL@10':>8}{'LL@20':>8}{'LL@30':>8}{'Q→80%':>14}{'Q→90%':>14}")
    for pol in wide.index.get_level_values(0).unique():
        m = wide.xs(pol).loc[students]
        stats = {0.8: [], 0.9: []}
        for _ in range(n_boot):
            idx = rng.choice(len(students), len(students))
            c = m.iloc[idx].mean().to_numpy()
            zero, fl = m.iloc[idx][0].mean(), full.loc[students].iloc[idx].mean()
            for f in stats:
                stats[f].append(questions_to_fraction(c, zero, fl, f))
        c = m.mean()
        ci = {f: np.nanpercentile(v, [2.5, 97.5]) for f, v in stats.items()}
        pt = {f: questions_to_fraction(c.to_numpy(), c[0], full.mean(), f) for f in stats}
        fmt = lambda f: f"{pt[f]:.0f} [{ci[f][0]:.0f},{ci[f][1]:.0f}]"
        print(f"{pol:<12}{c[5]:8.4f}{c[10]:8.4f}{c[20]:8.4f}{c[30]:8.4f}{fmt(0.8):>14}{fmt(0.9):>14}")


if __name__ == "__main__":
    main()
