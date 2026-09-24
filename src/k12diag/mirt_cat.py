"""Adaptive testing with a multidimensional IRT model.

Each student's posterior over theta is a Gaussian (Laplace approximation, prior N(0, I)),
refit after every answer. Policies that need expectations draw samples from it; with uniform
sample weights the expected-loss-reduction score is the same computation as on the 1D grid.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from k12diag.data import TestStudent
from k12diag.evaluate import brier, log_loss
from k12diag.irt import default_device
from k12diag.mirt import MIRT


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def laplace_posterior(
    A: np.ndarray, d: np.ndarray, y: np.ndarray, theta0: np.ndarray | None = None, iters: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """MAP and covariance of theta given responses y to items with loadings A (n, k), offsets d (n,)."""
    k = A.shape[1]
    theta = np.zeros(k) if theta0 is None else theta0.copy()
    for _ in range(iters):
        p = _sigmoid(A @ theta - d)
        grad = A.T @ (y - p) - theta
        H = (A * (p * (1 - p))[:, None]).T @ A + np.eye(k)
        step = np.linalg.solve(H, grad)
        theta += step
        if np.abs(step).max() < 1e-6:
            break
    p = _sigmoid(A @ theta - d)
    H = (A * (p * (1 - p))[:, None]).T @ A + np.eye(k)
    return theta, np.linalg.inv(H)


@dataclass
class Posterior:
    mu: np.ndarray
    cov: np.ndarray

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        L = np.linalg.cholesky(self.cov + 1e-9 * np.eye(len(self.mu)))
        return self.mu + rng.standard_normal((n, len(self.mu))) @ L.T


def _entropy_t(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log(p) + (1 - p) * torch.log1p(-p))


class MRandom:
    name = "random"

    def select(self, post, Ps, model, candidates, rng):
        return int(rng.choice(candidates))


class MDOptimal:
    """Bayesian D-optimal: maximize the determinant gain of the posterior precision, p(1-p) a' Sigma a."""

    name = "dopt"

    def select(self, post, Ps, model, candidates, rng):
        A = model.A[candidates]
        p = _sigmoid(A @ post.mu - model.d[candidates])
        gain = p * (1 - p) * np.einsum("qk,kl,ql->q", A, post.cov, A)
        return int(candidates[np.argmax(gain)])


class MExpectedLossReduction:
    """Summed mutual information between the candidate and every reference item, over posterior samples."""

    name = "elr"

    def __init__(self, reference_items: np.ndarray, device: torch.device | None = None):
        self.dev = device or default_device()
        self.ref = torch.tensor(reference_items, dtype=torch.long, device=self.dev)

    def scores(self, Ps: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        P = torch.tensor(Ps, dtype=torch.float32, device=self.dev)  # (S, n_items)
        R = P[:, self.ref]
        S = P.shape[0]
        current = _entropy_t(R.mean(0)).sum()
        Pc = P[:, torch.tensor(candidates, dtype=torch.long, device=self.dev)]  # (S, Q)
        expected = torch.zeros(len(candidates), device=self.dev)
        for lik in (Pc, 1 - Pc):
            p_y = lik.mean(0)  # (Q,)
            post_w = lik / (S * p_y)  # (S, Q), columns sum to 1
            expected += p_y * _entropy_t(post_w.T @ R).sum(1)
        return (current - expected).cpu().numpy()

    def select(self, post, Ps, model, candidates, rng):
        return int(candidates[np.argmax(self.scores(Ps, candidates))])


@dataclass
class MTrajectory:
    student: int
    policy: str
    items: list[int] = field(default_factory=list)
    answers: list[int] = field(default_factory=list)
    log_loss: list[float] = field(default_factory=list)
    brier: list[float] = field(default_factory=list)
    target_p: list[np.ndarray] = field(default_factory=list)


def item_probs(model: MIRT, theta_samples: np.ndarray) -> np.ndarray:
    """P(correct) for every sample and item, shape (S, n_items)."""
    return _sigmoid(theta_samples @ model.A.T - model.d[None, :])


def run_student_mirt(
    s: TestStudent, policy, model: MIRT, max_questions: int, rng: np.random.Generator, n_samples: int = 1000
) -> MTrajectory:
    answer_of = dict(zip(s.pool_items.tolist(), s.pool_y.tolist()))
    remaining = list(s.pool_items)
    k = model.k
    post = Posterior(np.zeros(k), np.eye(k))
    asked, ys = [], []
    traj = MTrajectory(student=s.student, policy=policy.name)

    for t in range(max_questions + 1):
        Ps = item_probs(model, post.samples(n_samples, rng))
        p = Ps[:, s.target_items].mean(0)
        traj.log_loss.append(log_loss(p, s.target_y))
        traj.brier.append(brier(p, s.target_y))
        traj.target_p.append(p)
        if t == max_questions or not remaining:
            break
        q = policy.select(post, Ps, model, np.array(remaining), rng)
        remaining.remove(q)
        asked.append(q)
        ys.append(answer_of[q])
        mu, cov = laplace_posterior(model.A[asked], model.d[asked], np.array(ys, dtype=float), post.mu)
        post = Posterior(mu, cov)
        traj.items.append(q)
        traj.answers.append(ys[-1])
    return traj


def full_information_mirt(
    s: TestStudent, model: MIRT, rng: np.random.Generator, n_samples: int = 1000
) -> tuple[float, np.ndarray]:
    mu, cov = laplace_posterior(model.A[s.pool_items], model.d[s.pool_items], s.pool_y.astype(float))
    p = item_probs(model, Posterior(mu, cov).samples(n_samples, rng))[:, s.target_items].mean(0)
    return log_loss(p, s.target_y), p
