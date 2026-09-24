"""Unidimensional 2PL IRT fit by marginal maximum likelihood on a fixed ability grid.

P(correct | theta, item j) = sigmoid(a_j * (theta - b_j)), theta ~ N(0, 1).

Using a grid for theta keeps test-time inference exact: a student's posterior is
just a weight vector over grid points, updated by multiplying in each response.
"""

from dataclasses import dataclass

import numpy as np
import torch


def default_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_grid(n_points: int = 81, lim: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(-lim, lim, n_points)
    log_prior = -0.5 * theta**2
    log_prior -= np.logaddexp.reduce(log_prior)
    return theta, log_prior


@dataclass
class IRT2PL:
    a: np.ndarray  # discrimination, shape (n_items,)
    b: np.ndarray  # difficulty, shape (n_items,)
    theta: np.ndarray  # grid, shape (G,)
    log_prior: np.ndarray  # shape (G,)

    def p_correct_grid(self) -> np.ndarray:
        """P(correct) for every grid point and item, shape (G, n_items)."""
        z = self.a[None, :] * (self.theta[:, None] - self.b[None, :])
        return 1.0 / (1.0 + np.exp(-z))


def fit_2pl(
    student: np.ndarray,
    item: np.ndarray,
    y: np.ndarray,
    n_students: int,
    n_items: int,
    n_grid: int = 61,
    steps: int = 100,
    log_a_sd: float = 0.5,
    device: torch.device | None = None,
    verbose: bool = False,
) -> IRT2PL:
    """Fit item parameters by maximizing the marginal likelihood of each student's responses.

    Full-batch L-BFGS rather than Adam: Adam's per-parameter step normalization moves
    low-data items at the same rate as well-determined ones, so they never settle.
    """
    theta_np, log_prior_np = make_grid(n_grid)
    dev = device or default_device()
    theta = torch.tensor(theta_np, dtype=torch.float32, device=dev)
    log_prior = torch.tensor(log_prior_np, dtype=torch.float32, device=dev)
    s = torch.tensor(student, dtype=torch.long, device=dev)
    j = torch.tensor(item, dtype=torch.long, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)

    log_a = torch.zeros(n_items, requires_grad=True, device=dev)
    b = torch.zeros(n_items, requires_grad=True, device=dev)
    opt = torch.optim.LBFGS([log_a, b], lr=1.0, max_iter=steps, line_search_fn="strong_wolfe")
    sign = torch.where(yt > 0.5, 1.0, -1.0)[:, None]

    def closure():
        opt.zero_grad()
        z = torch.exp(log_a)[j, None] * (theta[None, :] - b[j, None])  # (rows, G)
        # log p(y | theta) = y * logsig(z) + (1 - y) * logsig(-z)
        ll_rows = torch.nn.functional.logsigmoid(sign * z)
        ll_students = torch.zeros(n_students, len(theta), device=dev).index_add_(0, s, ll_rows)
        marginal = torch.logsumexp(ll_students + log_prior[None, :], dim=1)
        # Priors: log a ~ N(0, log_a_sd^2), b ~ N(0, 3^2).
        penalty = 0.5 * (log_a**2).sum() / log_a_sd**2 + 0.5 * (b**2 / 9.0).sum()
        loss = -(marginal.sum() - penalty) / len(y)
        loss.backward()
        return loss

    opt.step(closure)
    loss = closure()
    if verbose:
        print(f"final nll/response {loss.item():.4f}")

    return IRT2PL(
        a=torch.exp(log_a).detach().cpu().numpy().astype(np.float64),
        b=b.detach().cpu().numpy().astype(np.float64),
        theta=theta_np,
        log_prior=log_prior_np,
    )
