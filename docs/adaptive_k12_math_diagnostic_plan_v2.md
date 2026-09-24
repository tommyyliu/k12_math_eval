# Adaptive K–12 Math Diagnostic: Plan v2

## Key Question

**How few adaptively chosen math problems are needed to accurately predict how a student will perform on a broad distribution of unseen K–12 math problems?**

At step $t$, given response history $H_t$, the system estimates

$$
P(\text{correct} \mid \text{problem}, H_t)
$$

for unseen problems, and chooses the next problem expected to most improve those predictions:

$$
q^* = \arg\max_q \; \mathbb{E}\left[ L(H_t) - L(H_{t+1}) \mid q \right]
$$

The central empirical curve is

$$
\text{questions administered} \longrightarrow \text{held-out predictive loss}
$$

### What changed from v1

- **Real data comes first.** v1 built the loop on synthetic students and brought in real data at step 10. v2 builds the whole loop on the Eedi dataset first. Synthetic students then *extend* real data instead of replacing it.
- **Synthetic realism is measured, not assumed.** A real-vs-synthetic discriminator scores each simulator and tells us what to fix.
- **The goal includes understanding the policy.** We want to see what the adaptive question sequences look like and whether a few problems carry most of the diagnostic value.
- **The evaluation protocol is spelled out:** dense held-out students, a zero-question baseline and an explicit target distribution.
- **New questions are in scope:** predicting a new question's diagnostic value from its content (Phase 2), and generating better questions (Phase 3). Together with selection, these are all parts of one goal: getting to know a student as efficiently as possible.

---

## Phase 0: Build the Loop on Real Data (Eedi)

### Data

Eedi NeurIPS 2020 Education Challenge: real students answering multiple-choice diagnostic maths questions, with the chosen option, correctness and a topic hierarchy.

- Confirm the license and terms of use before building on it.
- The Task 3/4 subset has question images and was designed for question selection. Start there and use the full Task 1/2 data for pretraining item parameters if needed.
- Missingness is not random: teachers assign the quizzes, so the items a student answered correlate with their curriculum and level. Keep this in mind whenever results depend on *which* items were seen.
- Students learn over time. Eedi answers can span months, so an answer used for selection and an answer used for scoring may reflect different skill levels. Keep each test student's query pool and target set within a limited time window, and check whether results change with the window size.

### Evaluation protocol

- **Split by student.** Item parameters are learned only from training students.
- **For each test student,** split their answered items into:
  - a **query pool** the policy may choose from, and
  - a fixed **target set** that is never selectable and is used only for scoring.
- Prefer test students with many answers. If needed, pick a dense student × item block (for example by greedy biclustering) so the query pool is not tiny or arbitrary.
- **Target distribution:** the target set is whatever the student was assigned, which is mostly near their level. Report this explicitly. If we later build our own bank, define the target distribution deliberately; otherwise items far from the student's level inflate accuracy.

### Metrics

- Primary: held-out **log loss** vs. questions asked, with bootstrap confidence intervals over students.
- Headline numbers:
  - **Questions to reach 80% / 90% of the full-information gain**, i.e. how many questions each policy needs to close most of the gap between the zero-question baseline and the full-information line.
  - **Area under the loss curve** over the first 20 questions.
- Secondary: Brier score, calibration (ECE / reliability plots), AUC.
- **Reference lines:**
  - **Zero-question baseline:** population prior only. Report every curve as improvement over it.
  - **Full-information line:** predictions from the student's entire query pool, the best any policy can do with the pool.

### Models (simplest first)

1. 1PL / 2PL IRT
2. Multidimensional IRT or matrix factorization, $k = 1 \dots 10$. Held-out loss vs. $k$ answers *"what is the effective dimensionality?"*
3. **Nominal-response model:** use *which* option was picked, not just correct/incorrect. Eedi distractors are designed around misconceptions, so a wrong answer can be very informative.
4. Richer models (problem embeddings, neural/sequence models) only if they beat the above.

### Selection policies

1. Random
2. Fixed stratified set (by topic / difficulty)
3. Difficulty-adaptive (closest to current ability estimate)
4. Maximum Fisher information
5. Expected predictive-loss reduction (the $q^*$ rule above)
6. Later: learned policies (BOBCAT / NCAT style)

### Stopping

Treat STOP as an action: stop when the expected loss reduction of the best next question falls below a threshold. Report the distribution of test lengths across students, not just the mean.

### Deliverable

The loss-vs-questions curves for every policy × model, plus the two analyses below.

**First question to answer:** Does adaptive selection reduce held-out error substantially faster than random or fixed testing, on real students? "Substantially" means: the adaptive policy reaches 80% of the full-information gain with clearly fewer questions than random, with non-overlapping confidence intervals.

### Status (first pass, 2026-09-22)

With a 1D 2PL model on 1,000 held-out students (details in `results/phase0_2pl/`):

- **Adaptive beats random:** Fisher / expected loss reduction reach 80% of the full-information gain in ~11–12 questions vs. ~23 for random, with non-overlapping CIs.
- **In line with published work:** AUC 0.764 after 10 questions, vs. 0.744 (IRT-Active) and 0.766 (BOBCAT's best) on the same data. Protocols differ, so this means "comparable", not "better".
- **The policy is nearly saturated for this model:** the 1D full-information ceiling is AUC 0.781. The remaining headroom is in the model: multiple dimensions and wrong-answer information.
- **Diagnostic value is spread evenly under 1D:** the top 10% of items hold 18% of the value, and value is ~entirely discrimination (Spearman 0.99). Finding "especially diagnostic" problems needs richer models.

**Dimensionality (MIRT via variational inference, k = 1–10):**

- The k=1 fit reproduces the grid 2PL (full-information AUC 0.781), so the two implementations agree.
- With every available answer, more dimensions keep helping, with diminishing returns: AUC 0.781 → 0.788 (k=3) → 0.792 (k=6) → 0.794 (k=10); log loss 0.558 → 0.545. Still improving slightly at k=10.
- With few *random* answers, extra dimensions barely help: at 30 answers +0.4 AUC points; at ≤5 answers ≈0, and k=10 is slightly worse. A handful of random questions can't pin down several skills.
- So there is real multidimensional structure, but it's modest, and random questions can't get at it quickly. Whether *adaptive* selection can reach it early is the next test.

**Multidimensional adaptive selection** (Laplace posterior; random / Bayesian D-optimal / expected loss reduction; `results/phase0_mirt/`):

- k=1 reproduces the grid results (D-optimal = Fisher: 80% of the gain in 11 questions), so the implementations agree.
- **Crossover.** Paired log-loss difference, 6D vs 1D (both D-optimal): worse at 3 questions (+0.005, significant), even at 5–10, better at 20 (−0.005) and 30 (−0.006), both significant. Extra dimensions cost early and pay off after ~10–15 questions.
- **Headline:** 30 adaptive questions under the 6D model reach log loss 0.5578, the same as the 1D model with *all* ~156 available answers.
- Expected loss reduction beats D-optimal early (3 questions, significant) and loses later (20–30). A coarse-to-fine policy may get the best of both.
- "Questions to 80% of the gain" gets *larger* with k because the full-information ceiling drops; compare absolute loss across k, not this ratio.

**Wrong answers (nominal-response model; `results/nominal/`):**

- Seeing the chosen option, rather than just right/wrong, adds only ~0.1–0.25 AUC points (k=6: 0.750 → 0.752 at 10 answers, 0.773 → 0.775 at 30). No confidence intervals yet.
- Why: which wrong answer a student picks is a stable trait (split-half r = 0.58), but it correlates 0.69 with overall accuracy. Stronger students pick the plausible distractor; weaker students scatter. So it mostly re-measures general ability.
- A low-dimensional model can't represent specific misconceptions; misconception-level structure would need item-specific or topic-specific modeling.

**Next:** coarse-to-fine / model-averaged policy; per-item diagnostic value under 6D; calibration; paired CIs for the nominal gains.

---

## Phase 0 Analyses: Sequencing and Diagnostic Value

### A. What do the adaptive sequences look like?

A deterministic adaptive policy is a **decision tree**: every student gets the same first question, and the answer decides the branch.

- Draw the top levels of the tree (first 3–4 questions) with each node's topic, difficulty and option-level branching (which *wrong* answer sends you where).
- How quickly do paths diverge? How many distinct paths exist at depth 5 or 10?
- Does the policy begin broad (coarse ability placement) and then narrow into topic-specific probes?
- Compare with the fixed/stratified baselines. Where does adaptivity pay off, and at what step?

### B. Do some problems have much better diagnostic value than others?

Diagnostic value depends on context: a question is valuable *given what is already known*. So measure it several ways:

| Measure | What it captures |
|---|---|
| **First-question value**: expected loss reduction when asked with no history | Value in a vacuum |
| **Average marginal value**: mean loss reduction when selected mid-sequence | Value in context |
| **Selection frequency** under the adaptive policy | How often the policy relies on it |
| **Ablation**: remove the top-$k$ items from the bank and re-run the curve | Whether they are replaceable |

Then:

- Plot the distribution of diagnostic value across items. Is it heavy-tailed (a small core does most of the work) or flat?
- Describe the high-value items: difficulty, discrimination, topic, number of meaningful distractors, loadings on multiple dimensions.
- Test whether a small **core set** chosen from high-value items, used as a fixed test, comes close to the adaptive policy's performance.

---

## Phase 1: Synthetic Students, Measured by a Discriminator

### Simulating students

- Build naturalistic student profiles, as in v1: working style, strengths/weaknesses, misconceptions, uneven schooling. Avoid demographic proxies for ability.
- Use several LLM families and capability levels.
- Run simulated students on **Eedi items** (multimodal model or OCR'd text) so their answers can be compared directly with real students on the same questions.
- Try several ways of giving a student **persistent state**:
  - independent calls from the profile alone (baseline, likely unrealistic);
  - earlier questions and answers kept in context;
  - profiles tied to explicit latent traits and misconception lists.

### The discriminator

Train classifiers to tell real students from synthetic ones. Held-out AUC is the realism score: **0.5 = indistinguishable.**

**Matched comparisons.** For each real student, generate a synthetic student who answers *exactly the same items*. Otherwise the discriminator learns missingness patterns instead of behavior.

**Discriminators, from interpretable to powerful:**

1. Logistic regression on summary features: accuracy, response to difficulty, person-fit statistics ($l_z$, Guttman errors), distractor-choice entropy, consistency across related items.
2. Gradient-boosted trees on richer hand-built features.
3. A set/sequence model over (item, chosen option) pairs, which may find signals we did not think of.

**Using it to improve simulators.**

- Feature importance and misclassified examples show *how* synthetic students give themselves away. For example: too consistent, wrong answers that aren't the misconception distractors, errors that don't depend on difficulty.
- Each fix targets a specific tell, then we re-score.

**Guarding against Goodhart's law.**

- Retrain the discriminator every round. Keep a **final held-out discriminator** (different architecture and data) that is never used for tuning.
- Also check **population-level** realism, since realistic individuals can still come in the wrong mix: compare distributions of IRT ability, dimensionality and item difficulties (for example with an MMD two-sample test).
- Also check **downstream** realism: item parameters fitted on synthetic data should match those from real data, and policies built in simulation should rank the same way on real students.

### Deliverable

A leaderboard of simulator variants × realism scores, and a list of the tells that remain.

---

## Phase 2: How Much Will a New Question Tell Us About This Student?

The expected information from a question, for a given student, has two parts: the question's parameters (difficulty, discrimination, skill loadings, how attractive each distractor is) and the student's current posterior. The posterior part is solved, so the question reduces to **predicting a question's parameters, and our uncertainty about them, from its content alone**.

### Setup on Eedi

- **Question content:** extract text, answer options and diagrams from the 948 Task 3/4 question images with a vision model.
- **Split by question:** hold out ~20% of questions, fit the model on the rest, and predict the held-out questions' parameters from content.

### Predictors, simplest first

1. Topic averages
2. Regression on text / image embeddings
3. An LLM judging difficulty and how attractive each distractor is
4. Simulated students (Phase 1) answering the new question, fitted like real responses

### A more direct alternative

Instead of predicting parameters or embeddings, predict the quantity we care about directly: **does knowing a student's answer to this question improve predictions of their answers to other questions?** Concretely, predict the new question's co-response pattern (how its answers relate to answers on existing questions), or regress realized loss reduction on content directly. Compare this with the parameter route. The parameter route's advantage is that combining with a student's posterior gives per-student value for free, and a single "value" score doesn't.

### Evaluation

- Correlation between predicted and fitted parameters, per parameter.
- **What really counts:** the adaptive test's loss curve using *predicted* parameters for the held-out questions vs. *fitted* ones.
- Per-student: does predicted information gain match realized loss reduction for that student?

### Things to handle

- **Parameter uncertainty lowers value.** A question with uncertain parameters is worth less than a calibrated one with the same point estimate. Selection should integrate over parameter uncertainty.
- **Explore/exploit:** asking an uncalibrated question also teaches us about the question.
- This also gives the synthetic students a concrete job: their value is how well they calibrate new questions.

---

## Phase 3: Generating Better Questions

**"Better" = higher expected information gain** for a target student or population, while remaining valid: one correct answer, clear wording, appropriate level.

- **Reward model:** the Phase 2 predictor scores generated questions.
- **Per-student generation:** rather than choosing from a bank, generate the question that best separates the leading hypotheses about *this* student (e.g. misconception A vs. B). The multidimensional and nominal-response models describe what that uncertainty is.
- **Distractors are the likely lever:** wrong answers designed around specific misconceptions turn a single response into evidence about *which* misconception. Eedi's 2024 Kaggle "Mining Misconceptions" data labels distractors with misconceptions, which may help (check overlap with our questions).

**Risks:**

- **Reward hacking:** the generator will exploit errors in the predictor. Real student responses are the only real validation. Plan for a pilot with real students; synthetic students are an intermediate check at best.
- **Construct validity:** confusing wording, trick questions or heavy reading load can look highly discriminating because they separate strong readers or careful students, not math ability. Check that high-value questions are discriminating *for the math*: expert/LLM review, and loadings on the intended dimension rather than a generic "carefulness" one.

---

## Phase 4: Use Synthetic Data Where Real Data Is Missing

- **Broader problem bank:** add problems from released standardized tests and other high-quality sources across K–12. Store text, answer/scoring rule, response type, source, nominal grade/topic (as metadata, not ground truth), embedding and distractor rationale where available.
- **Stress tests:** vary profile generators, simulators, bank composition and difficulty mix. A policy that only works under one simulator is exploiting simulation artifacts.
- **Weight simulators by realism:** use Phase 1 scores to drop or downweight simulators and profiles that are easy to tell apart from real students.

---

## Phase 5: Better Policies and Real-World Cost

- Divide diagnostic value by expected time or effort per problem.
- Learned selection policies trained on real data plus realism-weighted synthetic data.
- Policies over the combined bank, validated on held-out real students.

---

## Longer-Term Questions

- How many questions are needed for useful predictive accuracy, and how much does adaptivity cut that?
- How does the needed test length vary across students?
- Which kinds of problems are most diagnostic, and why?
- What is the effective latent dimensionality of K–12 math performance, beyond overall ability?
- How far can synthetic students get on the discriminator, and what are the last tells to go?
- Do more capable LLM simulators produce more human-like errors?
- How well can a question's diagnostic value be predicted before anyone answers it?
- Can generated questions beat the best existing ones, on real students?
- What makes a question diagnostic: its distractors, its difficulty relative to the student, or the skills it combines?
- How large does the problem bank need to be?

## References

- Eedi NeurIPS 2020 Education Challenge (dataset and Task 4: personalized question selection)
- Ghosh & Lan, *BOBCAT: Bilevel Optimization-Based Computerized Adaptive Testing*, IJCAI 2021
- Zhuang et al., *Fully Adaptive Framework: Neural Computerized Adaptive Testing for Online Education* (NCAT), AAAI 2022
- Zhuang et al., *A Bounded Ability Estimation for Computerized Adaptive Testing* (BECAT), NeurIPS 2023
- Liu et al., *Survey of Computerized Adaptive Testing: A Machine Learning Perspective*, 2024
- Eedi, *Mining Misconceptions in Mathematics* (Kaggle, 2024)
