# Phase 0 Follow-ups: Plan

Closes out Phase 0 (see `adaptive_k12_math_diagnostic_plan_v2.md`, "Status") around four questions:

1. **Wrong answers.** How much information is in *which* wrong answer a student picks, and where is it?
2. **Adaptivity.** What does adaptive selection actually do that random selection doesn't?
3. **Information per question.** How much does each question tell us, and in which situations?
4. **Test length.** Do some students need more questions than others, and which ones?

These replace the "Next" list in the v2 status. Calibration and "analysis B under 6D" are folded in below. Coarse-to-fine is dropped: the gap it targets (~0.005 log loss) is too small to matter.

**Scope:** analysis scripts plus small, tested additions to `src/k12diag` (new policies, per-step recording). The data, split and fitted models stay as they are.

**Models:** questions 2–4 use the 1D 2PL grid model (exact posterior, interpretable, fast). The headline numbers are then repeated under 6D MIRT. Question 1 uses the nominal-response models (k = 1, 3, 6) that are already fitted.

---

## 0. Setup on the GPU machine

`data/` and `results/` are gitignored, so a fresh clone needs data plus regenerated baseline results.

1. `uv sync`, then check `uv run python -c "import torch; print(torch.cuda.is_available())"`. On Windows, PyPI's torch wheel is CPU-only. Point uv at the PyTorch CUDA index (`[tool.uv.sources]` / `[[tool.uv.index]]` in `pyproject.toml`) if needed.
2. **Make `default_device()` prefer CUDA** (`src/k12diag/irt.py`): order `cuda` → `mps` → `cpu`. Right now it checks only MPS, so on a PC everything silently runs on CPU.
3. Put the Eedi NeurIPS 2020 data under `data/raw/data/` with the same layout as before (`train_data/train_task_3_4.csv`, `metadata/answer_metadata_task_3_4.csv`, `metadata/question_metadata_task_3_4.csv`, `metadata/subject_metadata.csv`). Confirm the license and terms of use (still an open item from the v2 plan).
4. Regenerate the baseline results, in this order:
   ```
   uv run python scripts/run_phase0.py            # 2PL fit, 1D policy curves
   uv run python scripts/run_dimensionality.py    # MIRT k = 1..10
   uv run python scripts/run_phase0_mirt.py       # MIRT adaptive, k = 1, 3, 6
   uv run python scripts/run_nominal.py           # nominal-response models
   uv run python scripts/analyze_phase0.py
   uv run pytest
   ```
5. **Sanity check** against the v2 status numbers: 1D full-information AUC ≈ 0.781; Fisher/ELR reach 80% of the gain in ~11–12 questions vs ~23 for random; 6D full-information log loss ≈ 0.55. Small differences from CUDA vs MPS numerics are fine. Large ones mean something differs (data version, split) and must be resolved before going further.

**Where the GPU helps:** MIRT/nominal fitting and the torch ELR scoring in `mirt_cat.py`. The 1D grid code is numpy and runs one student at a time. If it's slow, parallelize over students with `multiprocessing` (students are independent; give each worker its own seeded RNG) rather than porting it to torch.

---

## 1. Wrong-answer information

Current result: seeing the chosen option adds ~0.1–0.25 AUC points, measured with random question order, no CIs.

Script: `scripts/analyze_wrong_answers.py`. Output: `results/phase0_followups/wrong_answers/`.

### 1a. Paired confidence intervals

- Change `run_nominal.py` to also save **per-student** log loss and target predictions for every (k, n, variant). It currently keeps only aggregates.
- For each k ∈ {1, 3, 6} and n ∈ {1, 3, 5, 10, 20, 30, all}: per-student difference `nominal_opt − nominal_rw` in log loss. Report the mean with a 95% bootstrap CI over students (2,000 resamples). Also report the AUC difference with a bootstrap CI (resample students, recompute pooled AUC).
- Also `nominal_rw − binary`, to check that the nominal model with right/wrong only is not worse than MIRT. If it is, the option gain is partly offset by a worse fit, and the comparison should say so.

### 1b. Which items carry the wrong-answer information?

Per item j, under the k=6 nominal model with θ ~ N(0, I) (Monte Carlo, ~4,000 samples):

- `I(option; θ)` and `I(correct; θ)`, where I(X; θ) = H(E_θ[P(X|θ)]) − E_θ[H(P(X|θ))].
- **Extra information from the option** = `I(option; θ) − I(correct; θ)`, which is ≥ 0.
- Also compute it at a typical mid-test posterior: θ samples from the Laplace posterior after 10 random answers, averaged over test students. Information at the prior can differ from information in context.

Report:
- The distribution across items. Is it heavy-tailed (a few items with well-designed distractors) or flat?
- Top 20 items: topic, difficulty, number of distractors actually chosen (options with ≥ 5% of wrong answers), distractor pick rates.
- **Model-free check:** for each item, among students who got it wrong, does *which* distractor they chose predict their accuracy on their other answers? Report the per-item effect size (η² from a one-way ANOVA of other-item accuracy on chosen distractor, with a permutation p-value). Compare the ranking with the model-based one (Spearman).

### 1c. Which students benefit?

Split test students into quintiles by full-information ability (1D posterior mean from the whole pool). Report the 1a difference with CIs per quintile, at n = 10 and n = 30. Hypothesis: weaker students gain more because they give more wrong answers.

### 1d. Does adaptive selection find the informative distractors?

So far the option gain was measured with random questions. An adaptive policy that knows about options might pick items whose distractors are informative.

- New module `src/k12diag/nominal_cat.py`: adaptive loop for the nominal model (Laplace posterior via `laplace_nominal`, posterior samples as in `mirt_cat.py`).
- **Option-aware ELR:** summed mutual information between the candidate's *option* (4 outcomes instead of 2) and the correctness of each reference item. This generalizes `MExpectedLossReduction`: loop over options instead of {correct, wrong}. Implement it in torch.
- Compare, k=6, n = 1..30:

  | Variant | Selects using | Observes |
  |---|---|---|
  | A | binary ELR | right/wrong |
  | B | binary ELR | option |
  | C | option-aware ELR | option |

  B − A = the option's value under the current policy. C − B = the value of *selecting for* distractors. Paired CIs as in 1a.
- Cost: one Laplace fit per step. Start with 200 students; go to 1,000 if the run time allows.

---

## 2. What adaptive selection does better than random

Script: `scripts/analyze_adaptivity.py`. Output: `results/phase0_followups/adaptivity/`.

### 2a. Policy ablation

Each new policy isolates one mechanism. All of them choose only from the student's pool, like the existing ones.

| Policy | Mechanism | Status |
|---|---|---|
| `random` | baseline | exists |
| `max_a` | highest discrimination; ignores the student | new |
| `static_value` | fixed ranking by first-question value (ELR at the prior); ignores the student | new |
| `difficulty` | predicted P(correct) closest to 0.5; ignores discrimination | exists |
| `fisher` | discrimination × difficulty matching | exists |
| `elr` | expected loss reduction over the bank | exists |

`static_value` is nearly `max_a` in 1D (Spearman 0.99). Keep both anyway: they diverge under 6D.

Add `MaxDiscriminationPolicy` and `StaticRankPolicy` to `policies.py` (tests: each picks the expected item on a toy bank). For 6D, add the MIRT counterparts to `mirt_cat.py`: `max_a` = largest ‖A_j‖, `difficulty` = predicted P closest to 0.5 using posterior samples, `static_value` = ELR at the N(0, I) prior.

**Headline metric:** area under the loss curve for questions 1–20, per student. **Share of the random→ELR gap each policy closes** = (AULC_random − AULC_policy) / (AULC_random − AULC_elr), with a paired bootstrap CI. Run under 1D and 6D.

How to read it: if `max_a` closes most of the gap, adaptivity mostly means "ask good items", and a fixed test would do nearly as well. If `difficulty` closes most of it, it's targeting. If only `fisher`/`elr` close it, it's the combination.

### 2b. What the policies do differently

For each policy, at each step:
- Distribution of the chosen item's **predicted P(correct)** at the moment it's asked, and the realized correct rate. Random probably spends many questions on near-certain items.
- Discrimination of asked items.
- **Topic coverage:** entropy of `area` among the first 10 asked items, and the fraction of the target set's areas covered by then.
- Posterior SD of θ after t questions (1D).

Also describe the pools: pools are teacher-assigned and mostly near the student's level, which limits how much difficulty matching *can* help. Report the pool's predicted-P distribution per student, so the ablation can be read against it.

### 2c. Where the adaptive advantage is largest

Per-student AULC difference (random − ELR) against full-information ability, pool size and pool difficulty spread (SD of b in the pool). Binned means with CIs. Question: is the adaptive advantage concentrated in students at the extremes, or students with wide pools?

---

## 3. Information per question, and when

Same script as 2. Output: `results/phase0_followups/per_step/`.

### 3a. Per-step records

Extend `run_student` (1D) with an optional per-step log: step t, item, answer, predicted P(correct) before asking, posterior mean and SD before and after, the model's **expected** target-loss reduction for the chosen item (ELR scored against the student's target set, computed for logging only, never for selection), and the **realized** target-loss drop. Rerun `random` and `elr` for all 1,000 test students (cheap in 1D). Save as `per_step.parquet`.

Also record the posterior entropy drop over θ, in nats. This is a model-internal measure of information that is much less noisy than the realized loss drop.

### 3b. Where the information comes from

Realized drop and θ-entropy drop, as binned means with CIs, by:
- step t (1, 2–3, 4–10, 11–30)
- the item's predicted P(correct) (deciles)
- surprise: |answer − predicted P|
- student ability (quintiles)
- whether the item's `area` appears in the student's target set

Then a linear regression of the realized drop on these features, with CIs from a student-level bootstrap, to separate them (e.g. whether "early questions are worth more" is really "early questions are closer to P = 0.5").

### 3c. Calibration of expected value

Bin steps by expected target-loss reduction. Plot the mean realized drop per bin against the expected. On the diagonal means the model's sense of what a question is worth can be trusted, which matters for stopping (question 4) and for Phase 2.

### 3d. Calibration of predictions (from the old "Next" list)

Reliability diagrams and ECE (15 equal-mass bins) of the target-set predictions at n ∈ {0, 5, 10, 30, full}, for 1D random/ELR, 6D random/D-opt/ELR, and nominal. This needs target predictions at those n, which the saved 1D trajectories currently drop. Keep those snapshots when rerunning in 3a.

---

## 4. Do some students need more questions?

Script: `scripts/analyze_test_length.py`. Output: `results/phase0_followups/test_length/`.

A single student's realized loss curve is noisy (target sets are ~30% of ≥80 answers), so the primary measures are model-based. The realized curves are used to check them.

### 4a. Stopping rules (1D, ELR selection)

- **Expected-value stop:** stop when the best candidate's ELR score, per reference item, falls below τ. This is STOP from the v2 plan. Use 3–4 values of τ spanning the observed range of scores at t = 5–30.
- **Precision stop:** stop when the posterior SD of θ < σ*, σ* ∈ {0.5, 0.4, 0.3}.
- Run up to 60 questions or until the pool is exhausted. Report the **full distribution** of test lengths (histogram, quantiles) per rule, not just the mean. Report the fraction censored by pool size separately; those students hit the pool limit, not the rule.

### 4b. Which students need more questions?

Relate test length to:
- full-information ability (expect the extremes to need more, because the bank has few extreme items)
- person-fit l_z on the full pool (inconsistent response patterns)
- pool size and pool difficulty spread
- **profile unevenness** under 6D: the norm of the student's full-information 6D posterior mean after projecting out the general-ability direction (the first principal direction of training students' θ)

Binned means plus one regression. For the rule-based lengths, also check that the ordering of students is stable across τ / σ*.

### 4c. Checks against real outcomes

- **Do early stoppers really need fewer?** Bin students by stopping length under each rule. For each bin, measure the realized target-loss improvement from the stopping point to t = 30. Early stoppers should gain little after stopping. If they gain as much as everyone else, the rule is stopping too early for them.
- **Are per-student differences real?** Split each student's target set in half at random. Compute "questions to 80% of this student's own gain" on each half separately, and report the split-half correlation (Spearman-Brown corrected) across students. If it's near zero, per-student test-length differences in the realized data are noise, and only the model-based measures can speak to the question.

---

## Deliverables

- `results/phase0_followups/analysis.md`: one section per question, headline numbers with CIs, figures alongside.
- A new "Status (follow-ups)" section in `adaptive_k12_math_diagnostic_plan_v2.md`, in the same style as the existing one.
- Tests in `tests/` for the new policies, the per-item information calculation (e.g. extra information ≈ 0 for an item whose distractors don't depend on θ; I(option) ≥ I(correct)), option-aware ELR on a toy model (it should equal binary ELR when all wrong options share the same parameters), and `default_device()`.

## Suggested order

1. Setup and sanity check (section 0).
2. 3a per-step rerun (1D, cheap). It feeds 2b, 3 and 4.
3. 2a ablation, 1D then 6D.
4. 1a and 1c (only needs the `run_nominal.py` change), then 1b.
5. 4a–4c.
6. 1d: the most expensive and least certain piece.
7. Write-up.
