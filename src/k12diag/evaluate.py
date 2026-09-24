"""Run selection policies on held-out students and score predictions on their target sets."""

from dataclasses import dataclass, field

import numpy as np

from k12diag.data import TestStudent
from k12diag.policies import Policy


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


@dataclass
class Trajectory:
    student: int
    policy: str
    items: list[int] = field(default_factory=list)  # asked items, in order
    answers: list[int] = field(default_factory=list)
    log_loss: list[float] = field(default_factory=list)  # index t = after t questions
    brier: list[float] = field(default_factory=list)
    target_p: list[np.ndarray] = field(default_factory=list)  # predictions after t questions


def run_student(
    s: TestStudent,
    policy: Policy,
    P: np.ndarray,
    log_prior: np.ndarray,
    max_questions: int,
    rng: np.random.Generator,
) -> Trajectory:
    """Ask up to `max_questions` from the student's pool, updating a grid posterior after each answer."""
    answer_of = dict(zip(s.pool_items.tolist(), s.pool_y.tolist()))
    remaining = list(s.pool_items)
    log_w = log_prior.copy()
    P_target = P[:, s.target_items]
    traj = Trajectory(student=s.student, policy=policy.name)

    for t in range(max_questions + 1):
        w = np.exp(log_w - log_w.max())
        w /= w.sum()
        p = w @ P_target
        traj.log_loss.append(log_loss(p, s.target_y))
        traj.brier.append(brier(p, s.target_y))
        traj.target_p.append(p)
        if t == max_questions or not remaining:
            break
        q = policy.select(w, P, np.array(remaining), rng)
        remaining.remove(q)
        y = answer_of[q]
        log_w += np.log(np.clip(P[:, q] if y else 1.0 - P[:, q], 1e-12, None))
        traj.items.append(q)
        traj.answers.append(y)
    return traj


def full_information(s: TestStudent, P: np.ndarray, log_prior: np.ndarray) -> tuple[float, np.ndarray]:
    """Target log loss after observing the student's entire pool: the best any policy can do."""
    Pp = np.clip(P[:, s.pool_items], 1e-12, 1 - 1e-12)
    log_w = log_prior + (s.pool_y * np.log(Pp) + (1 - s.pool_y) * np.log(1 - Pp)).sum(1)
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    p = w @ P[:, s.target_items]
    return log_loss(p, s.target_y), p
