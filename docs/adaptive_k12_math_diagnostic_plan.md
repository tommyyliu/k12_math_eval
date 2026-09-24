# Adaptive K–12 Math Diagnostic: Key Question and Plan

> **Superseded** by [adaptive_k12_math_diagnostic_plan_v2.md](adaptive_k12_math_diagnostic_plan_v2.md). Kept temporarily for reference.


## Key Question

**How few adaptively chosen math problems are needed to accurately predict how a student will perform on a broad distribution of unseen K–12 math problems?**

Rather than defining success in terms of grade levels, progression years, or manually specified skill mastery, the core objective is predictive:

> Given a student's responses to a small number of selected problems, how well can we predict their responses to many other math problems they have not seen?

This turns the assessment problem into an adaptive prediction problem.

At step $t$, given response history $H_t$, the system estimates:

$$
P(\text{correct} \mid \text{problem}, H_t)
$$

for unseen problems.

The adaptive policy should choose the next problem that is expected to most improve those predictions, while minimizing total assessment cost.

A simple optimization target is:

$$
\min \; \mathbb{E}[T]
$$

subject to held-out predictive loss being below some desired threshold, where $T$ is the number of administered problems.

The central empirical curve is therefore:

$$
\text{number of questions asked}
\longrightarrow
\text{predictive accuracy on unseen problems}
$$

---

## High-Level Plan

### 1. Assemble a broad problem bank

Start with roughly **1,000–5,000 K–12 math problems** drawn from released standardized tests and other high-quality sources.

For each problem, store:

- problem text
- correct answer / scoring rule
- response type
- source
- nominal grade or difficulty, if available
- nominal topic, if available
- semantic embedding
- any useful source metadata

Grade and topic labels are metadata, not ground truth.

---

### 2. Generate diverse synthetic student profiles

Create naturalistic student descriptions rather than rigid parameter lists.

Useful dimensions might include:

- meticulous vs. rushed
- persistent vs. gives up quickly
- strong vs. weak working memory
- intuitive vs. procedural problem solving
- strong verbal reasoning
- strong spatial reasoning
- calculator dependence
- uneven schooling
- specific conceptual strengths and weaknesses
- common misconceptions
- comfort with unfamiliar notation
- tendency to overthink or answer impulsively

Include highly uneven and unusual profiles rather than only smooth grade-level progressions.

Avoid using demographic characteristics as direct proxies for mathematical ability.

---

### 3. Simulate student responses with multiple LLMs

For each simulated attempt, provide:

$$
\text{student profile} + \text{math problem}
\rightarrow
\text{student response}
$$

Record:

- the student's answer
- optional reasoning/work
- whether the answer is correct
- which simulator/model produced it

Use multiple LLM families and capability levels so the dataset does not merely reproduce one model's internal biases or ontology.

Later, compare which simulators best resemble real student response patterns.

---

### 4. Build a sparse student × problem response dataset

Construct a large response matrix:

$$
Y_{ij} =
\text{response of student } i
\text{ on problem } j
$$

The matrix does not need to be dense.

For an initial experiment:

- ~500 synthetic students
- ~1,000 problems
- ~100–200 answered problems per student

Hold out a substantial set of responses for evaluation.

---

### 5. Train a predictive student model

Given a student's observed responses, predict how they will perform on unseen problems:

$$
P(Y_{ij}=1 \mid H_i, x_j)
$$

where:

- $H_i$ is the student's observed response history
- $x_j$ represents the candidate problem

Start with simple models before adding complexity:

1. one-dimensional IRT-style baseline
2. multidimensional IRT
3. matrix factorization / latent student and problem vectors
4. models incorporating problem embeddings
5. richer neural or sequence models if justified

A useful side question is:

> What is the effective latent dimensionality of K–12 mathematical performance?

---

### 6. Make held-out predictive performance the primary metric

For each student, hide many problem responses.

Allow the diagnostic system to observe only a small number of adaptively selected responses.

Then evaluate its predictions on the held-out problems.

Possible metrics include:

- log loss
- Brier score
- calibration
- accuracy / AUC as secondary metrics

The main graph is:

$$
\text{held-out prediction loss}
\quad \text{vs.} \quad
\text{questions administered}
$$

---

### 7. Compare question-selection policies

Start with simple baselines:

1. random selection
2. grade/topic-stratified fixed selection
3. difficulty-adaptive selection
4. uncertainty / information-gain selection
5. expected predictive-loss reduction

The core adaptive rule is approximately:

$$
q^*
=
\arg\max_q
\mathbb{E}
\left[
L(H_t) - L(H_{t+1})
\mid q
\right]
$$

In words:

> Ask the problem whose possible responses are expected to most improve predictions across the rest of the problem universe.

Eventually, divide this value by estimated problem cost or completion time.

---

### 8. Allow adaptive stopping

The system should not be required to administer a fixed-length test.

Treat **STOP** as an action.

Stop when the expected value of another question becomes sufficiently small.

This lets test length naturally vary by student.

A younger or more internally consistent student may require fewer problems; an older, highly uneven, or unusual student may require more.

---

### 9. Stress-test the synthetic world

Before trusting results, vary:

- student-profile generation schemes
- LLM simulators
- simulator capability levels
- problem-bank composition
- problem difficulty distributions
- unusual or adversarial student profiles

A policy that works only under one simulator or one profile generator is probably exploiting artifacts of the simulation.

---

### 10. Bring in real student data

As soon as practical, compare the synthetic population against real student response patterns.

Use real data to:

- identify unrealistic simulated students
- cull or downweight unrealistic synthetic profiles
- measure simulator bias
- adjust the distribution of synthetic students
- test whether policies learned in simulation transfer to humans

Synthetic students are primarily a way to build and debug the system before enough real data exists.

---

## Initial MVP

A reasonable first experiment:

- **1,000 math problems**
- **500 synthetic students**
- **100–200 responses per student**
- several LLM-based student simulators
- one simple latent-factor prediction model
- random selection baseline
- one adaptive selection method
- large held-out response set

The first question to answer is simply:

> **Does adaptive problem selection reduce held-out prediction error substantially faster than random or fixed testing?**

If yes, the loop is working.

From there, improve the problem bank, student simulation, prediction model, and adaptive policy one component at a time.

---

## Longer-Term Questions

Once the basic loop works, the project can empirically investigate:

- How many questions are needed for useful predictive accuracy?
- How much does adaptivity reduce test length?
- How does required test length vary across students?
- How large does the problem bank need to be?
- Which kinds of problems are maximally diagnostic?
- How many latent dimensions are needed to model mathematical performance?
- How well do synthetic-student-trained policies transfer to real students?
- Do smarter LLM simulators generate more human-like error patterns?
- Can problem embeddings help generalize diagnostic value to new problems?
- At what point do additional questions provide negligible predictive improvement?

The ultimate goal is not a fixed "minimal worksheet," but a compact adaptive policy that learns enough about each individual student to predict their mathematical performance across a broad K–12 problem space.
