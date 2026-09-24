"""Question-selection policies over a grid posterior.

Each policy sees the current posterior weights over the ability grid and the
remaining candidate items, and returns the index of the next item to ask.
"""

import numpy as np


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


class Policy:
    name = "base"

    def select(self, w: np.ndarray, P: np.ndarray, candidates: np.ndarray, rng) -> int:
        raise NotImplementedError


class RandomPolicy(Policy):
    name = "random"

    def select(self, w, P, candidates, rng):
        return int(rng.choice(candidates))


class DifficultyPolicy(Policy):
    """Ask the item whose predicted P(correct) is closest to 0.5."""

    name = "difficulty"

    def select(self, w, P, candidates, rng):
        p = w @ P[:, candidates]
        return int(candidates[np.argmin(np.abs(p - 0.5))])


class FisherPolicy(Policy):
    """Maximum Fisher information at the posterior-mean ability."""

    name = "fisher"

    def __init__(self, model):
        self.model = model

    def select(self, w, P, candidates, rng):
        theta_hat = w @ self.model.theta
        a, b = self.model.a[candidates], self.model.b[candidates]
        p = 1.0 / (1.0 + np.exp(-a * (theta_hat - b)))
        return int(candidates[np.argmax(a**2 * p * (1 - p))])


class ExpectedLossReductionPolicy(Policy):
    """Ask the item expected to most reduce predictive log loss over a reference item set.

    Under the model, expected log loss on item j equals the entropy of its predictive
    P(correct). The reduction from asking q is the summed mutual information between
    Y_q and every Y_j in the reference set.
    """

    name = "elr"

    def __init__(self, reference_items: np.ndarray | None = None):
        self.reference_items = reference_items

    def scores(self, w, P, candidates):
        R = P if self.reference_items is None else P[:, self.reference_items]
        current = _entropy(w @ R).sum()
        Pc = P[:, candidates]  # (G, Q)
        expected = np.zeros(len(candidates))
        for y_prob in (Pc, 1.0 - Pc):
            joint = w[:, None] * y_prob  # (G, Q) unnormalized posterior after outcome
            p_y = joint.sum(0)  # (Q,)
            post = joint / p_y  # (G, Q)
            expected += p_y * _entropy(post.T @ R).sum(1)
        return current - expected

    def select(self, w, P, candidates, rng):
        return int(candidates[np.argmax(self.scores(w, P, candidates))])
