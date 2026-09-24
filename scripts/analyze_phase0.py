"""Phase 0 analyses: (A) what adaptive sequences look like, (B) which items carry diagnostic value.

Usage: uv run python scripts/analyze_phase0.py
"""

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from k12diag.data import load_question_subjects, load_task34
from k12diag.policies import ExpectedLossReductionPolicy

OUT = Path(__file__).resolve().parents[1] / "results" / "phase0_2pl"


def item_table(model) -> pd.DataFrame:
    df = load_task34()
    ids = df.drop_duplicates("item").set_index("item")["QuestionId"].sort_index()
    subj = load_question_subjects().set_index("QuestionId")
    t = pd.DataFrame({"QuestionId": ids.values, "a": model.a, "b": model.b}, index=ids.index)
    t["n_responses"] = df.groupby("item").size()  # all students, incl. test
    return t.join(subj, on="QuestionId")


def policy_tree(model, depth: int, items: pd.DataFrame) -> list[str]:
    """Unrestricted expected-loss-reduction policy over the whole bank, branching on each answer."""
    P = model.p_correct_grid()
    bank = items.index.to_numpy()
    pol = ExpectedLossReductionPolicy(bank)
    lines = []

    def walk(log_w, asked, prefix, d):
        if d == depth:
            return
        w = np.exp(log_w - log_w.max())
        w /= w.sum()
        cand = np.setdiff1d(bank, asked)
        q = pol.select(w, P, cand, None)
        p = float(w @ P[:, q])
        it = items.loc[q]
        theta_hat = float(w @ model.theta)
        lines.append(
            f"{prefix}Q{it.QuestionId} [{it.area} / {it.subject}] a={it.a:.2f} b={it.b:+.2f} "
            f"P(correct)={p:.2f}  (ability est {theta_hat:+.2f})"
        )
        for y, tag in ((1, "✓"), (0, "✗")):
            lik = P[:, q] if y else 1 - P[:, q]
            lines.append(f"{prefix}  {tag}")
            walk(log_w + np.log(lik), np.append(asked, q), prefix + "    ", d + 1)

    walk(model.log_prior.copy(), np.array([], dtype=int), "", 0)
    return lines


def main():
    model = pickle.loads((OUT / "model.pkl").read_bytes())
    curves = pd.read_parquet(OUT / "curves.parquet")
    trajs = pickle.loads((OUT / "trajectories.pkl").read_bytes())
    saved = pickle.loads((OUT / "split.pkl").read_bytes())
    test, bank = saved["test"], saved["bank"]
    items = item_table(model).loc[bank]
    P = model.p_correct_grid()
    report = []

    # Loss curves
    fig, ax = plt.subplots(figsize=(7, 4.5))
    full = curves.groupby("student").full_info_log_loss.first().mean()
    for pol, g in curves.groupby("policy"):
        m = g.groupby("t").log_loss.mean()
        ax.plot(m.index, m.values, label=pol)
    ax.axhline(full, ls="--", c="gray", label="full information")
    ax.set_xlabel("questions asked")
    ax.set_ylabel("held-out log loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "loss_curves.png", dpi=150)

    # A1. Policy tree over the whole bank
    report += ["## A1. Unrestricted policy tree (whole bank, model-simulated answers)", ""]
    report += policy_tree(model, depth=3, items=items)

    # A2. Empirical sequences under pool restriction
    elr = [t for t in trajs if t["policy"] == "elr"]
    report += ["", "## A2. Empirical ELR sequences (restricted to each student's pool)", ""]
    for k in (1, 3, 5, 10):
        n_paths = len({tuple(t["items"][:k]) for t in elr if len(t["items"]) >= k})
        report.append(f"distinct paths at depth {k}: {n_paths} of {len(elr)} students")
    areas = items["area"]
    for step in (0, 1, 2, 4, 9, 19, 29):
        a = pd.Series([areas.loc[t["items"][step]] for t in elr if len(t["items"]) > step])
        top = ", ".join(f"{k} {v:.0%}" for k, v in a.value_counts(normalize=True).head(3).items())
        report.append(f"step {step + 1:>2}: top areas {top}")

    # B. Diagnostic value
    ref_scores = ExpectedLossReductionPolicy(bank).scores(
        np.exp(model.log_prior) / np.exp(model.log_prior).sum(), P, bank
    )
    items["first_q_value"] = ref_scores

    availability = pd.Series(np.concatenate([s.pool_items for s in test])).value_counts()
    chosen = pd.Series(np.concatenate([t["items"] for t in elr])).value_counts()
    items["elr_select_rate"] = (chosen / availability).reindex(items.index).fillna(0)

    # Realized marginal value: loss drop on the target set when an item is asked, under random order.
    drops = [
        (q, t["log_loss"][k] - t["log_loss"][k + 1])
        for t in trajs if t["policy"] == "random"
        for k, q in enumerate(t["items"])
    ]
    d = pd.DataFrame(drops, columns=["item", "drop"]).groupby("item")["drop"].agg(["mean", "count"])
    items["realized_marginal_value"] = d["mean"].where(d["count"] >= 30)

    v = items["first_q_value"].sort_values(ascending=False)
    share = v.cumsum() / v.sum()
    report += ["", "## B. Diagnostic value", ""]
    report.append(
        f"first-question value: top 1% of items hold {share.iloc[len(v) // 100 - 1]:.0%} of total, "
        f"top 10% hold {share.iloc[len(v) // 10 - 1]:.0%}; max/median ratio {v.iloc[0] / v.median():.1f}"
    )
    corr = items[["first_q_value", "elr_select_rate", "realized_marginal_value", "a", "b"]].corr(
        method="spearman"
    )
    report += ["", "Spearman correlations:", "```", corr.round(2).to_string(), "```"]
    cols = ["QuestionId", "area", "subject", "n_responses", "a", "b", "first_q_value", "elr_select_rate",
            "realized_marginal_value"]
    report += ["", "Top 15 items by first-question value:", "```",
               items.sort_values("first_q_value", ascending=False)[cols].head(15).round(3).to_string(), "```"]
    by_area = items.groupby("area")[["first_q_value", "a"]].median().sort_values("first_q_value", ascending=False)
    report += ["", "Median value by area:", "```", by_area.round(3).to_string(), "```"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(items["first_q_value"], bins=50)
    axes[0].set_xlabel("first-question value (nats, summed over bank)")
    axes[0].set_ylabel("items")
    axes[1].scatter(items["b"], items["a"], c=items["first_q_value"], s=8, cmap="viridis")
    axes[1].set_xlabel("difficulty b")
    axes[1].set_ylabel("discrimination a")
    fig.tight_layout()
    fig.savefig(OUT / "diagnostic_value.png", dpi=150)

    items.to_csv(OUT / "items.csv")
    (OUT / "analysis.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
