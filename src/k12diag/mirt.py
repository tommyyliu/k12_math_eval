"""Multidimensional (compensatory) IRT fit by variational inference.

P(correct | theta_i, item j) = sigmoid(A_j . theta_i - d_j), theta_i ~ N(0, I_k).

Each student gets a diagonal Gaussian posterior q(theta_i) = N(mu_i, diag(sigma_i^2)); item
parameters are point estimates with Gaussian priors. The same variational fit, with item
parameters frozen, gives posteriors for held-out students from any subset of their answers.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from k12diag.irt import default_device


@dataclass
class MIRT:
    A: np.ndarray  # (n_items, k) loadings
    d: np.ndarray  # (n_items,) difficulty offsets

    @property
    def k(self) -> int:
        return self.A.shape[1]


def _elbo_terms(mu, log_sigma, A, d, s, j, y, n_samples):
    """Expected log-likelihood per row (MC over q) and KL per student."""
    eps = torch.randn(n_samples, *mu.shape, device=mu.device)
    theta = mu + torch.exp(log_sigma) * eps  # (S, n_students, k)
    z = (theta[:, s, :] * A[j][None]).sum(-1) - d[j][None]  # (S, rows)
    sign = y * 2 - 1
    ll = F.logsigmoid(sign[None] * z).mean(0)  # (rows,)
    kl = 0.5 * (mu**2 + torch.exp(2 * log_sigma) - 2 * log_sigma - 1).sum(-1)  # (n_students,)
    return ll, kl


def fit_mirt(
    student: np.ndarray,
    item: np.ndarray,
    y: np.ndarray,
    n_students: int,
    n_items: int,
    k: int,
    steps: int = 2000,
    lr: float = 0.03,
    a_sd: float = 1.0,
    n_samples: int = 4,
    seed: int = 0,
    device: torch.device | None = None,
    verbose: bool = False,
) -> MIRT:
    dev = device or default_device()
    torch.manual_seed(seed)
    s = torch.tensor(student, dtype=torch.long, device=dev)
    j = torch.tensor(item, dtype=torch.long, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)

    A = (0.1 * torch.randn(n_items, k, device=dev)).requires_grad_()
    d = torch.zeros(n_items, device=dev, requires_grad=True)
    mu = torch.zeros(n_students, k, device=dev, requires_grad=True)
    log_sigma = torch.full((n_students, k), -1.0, device=dev, requires_grad=True)
    opt = torch.optim.Adam([A, d, mu, log_sigma], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    for step in range(steps):
        opt.zero_grad()
        ll, kl = _elbo_terms(mu, log_sigma, A, d, s, j, yt, n_samples)
        prior = 0.5 * (A**2).sum() / a_sd**2 + 0.5 * (d**2).sum() / 9.0
        loss = -(ll.sum() - kl.sum() - prior) / len(y)
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step % 500 == 0 or step == steps - 1):
            print(f"  k={k} step {step:5d}  -elbo/response {loss.item():.4f}")

    return MIRT(A=A.detach().cpu().numpy(), d=d.detach().cpu().numpy())


def infer_students(
    model: MIRT,
    student: np.ndarray,
    item: np.ndarray,
    y: np.ndarray,
    n_students: int,
    steps: int = 400,
    lr: float = 0.05,
    n_samples: int = 8,
    seed: int = 0,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior (mu, sigma) for students 0..n_students-1 given their observed rows, items frozen.

    Students with no observed rows get the prior.
    """
    dev = device or default_device()
    torch.manual_seed(seed)
    A = torch.tensor(model.A, dtype=torch.float32, device=dev)
    d = torch.tensor(model.d, dtype=torch.float32, device=dev)
    mu = torch.zeros(n_students, model.k, device=dev, requires_grad=True)
    log_sigma = torch.zeros(n_students, model.k, device=dev, requires_grad=True)
    if len(y) == 0:
        return mu.detach().cpu().numpy(), np.ones((n_students, model.k))

    s = torch.tensor(student, dtype=torch.long, device=dev)
    j = torch.tensor(item, dtype=torch.long, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam([mu, log_sigma], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        ll, kl = _elbo_terms(mu, log_sigma, A, d, s, j, yt, n_samples)
        (-(ll.sum() - kl.sum())).backward()
        opt.step()
    return mu.detach().cpu().numpy(), np.exp(log_sigma.detach().cpu().numpy())


def predict(
    model: MIRT,
    mu: np.ndarray,
    sigma: np.ndarray,
    student: np.ndarray,
    item: np.ndarray,
    n_samples: int = 256,
    seed: int = 0,
) -> np.ndarray:
    """Posterior predictive P(correct) for (student, item) rows, averaging over q(theta)."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((n_samples, len(student), model.k))
    theta = mu[student][None] + sigma[student][None] * eps
    z = (theta * model.A[item][None]).sum(-1) - model.d[item][None]
    return (1 / (1 + np.exp(-z))).mean(0)
