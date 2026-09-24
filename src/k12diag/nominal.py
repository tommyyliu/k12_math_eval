"""Multidimensional nominal-response model: which option a student picks, not just right/wrong.

P(option c | theta, item j) = softmax_c(B_jc . theta + c_jc), theta ~ N(0, I_k).

Observations are sets of options: {chosen option} when we see the choice, or {correct} /
{all wrong options} when we only see correctness. One likelihood, log sum_{c in S} P(c),
covers both, so the value of seeing *which* wrong answer was picked can be measured directly.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from k12diag.irt import default_device


@dataclass
class Nominal:
    B: np.ndarray  # (n_items, n_options, k)
    c: np.ndarray  # (n_items, n_options)
    correct: np.ndarray  # (n_items,) index of the correct option

    @property
    def k(self) -> int:
        return self.B.shape[2]

    def option_probs(self, theta: np.ndarray, items: np.ndarray) -> np.ndarray:
        """P(option) for theta samples (S, k) and items, shape (S, len(items), n_options)."""
        logits = np.einsum("sk,jck->sjc", theta, self.B[items]) + self.c[items][None]
        logits -= logits.max(-1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(-1, keepdims=True)

    def p_correct(self, theta: np.ndarray, items: np.ndarray) -> np.ndarray:
        """P(correct) for theta samples and items, shape (S, len(items))."""
        P = self.option_probs(theta, items)
        return np.take_along_axis(P, self.correct[items][None, :, None], axis=2)[..., 0]


def fit_nominal(
    student: np.ndarray,
    item: np.ndarray,
    option: np.ndarray,
    correct: np.ndarray,
    n_students: int,
    n_items: int,
    k: int,
    n_options: int = 4,
    steps: int = 2000,
    lr: float = 0.03,
    b_sd: float = 1.0,
    n_samples: int = 4,
    seed: int = 0,
    device: torch.device | None = None,
    verbose: bool = False,
) -> Nominal:
    """Variational fit, as in mirt.fit_mirt, with a softmax over options in place of the logistic."""
    dev = device or default_device()
    torch.manual_seed(seed)
    s = torch.tensor(student, dtype=torch.long, device=dev)
    j = torch.tensor(item, dtype=torch.long, device=dev)
    o = torch.tensor(option, dtype=torch.long, device=dev)

    B = (0.1 * torch.randn(n_items, n_options, k, device=dev)).requires_grad_()
    c = torch.zeros(n_items, n_options, device=dev, requires_grad=True)
    mu = torch.zeros(n_students, k, device=dev, requires_grad=True)
    log_sigma = torch.full((n_students, k), -1.0, device=dev, requires_grad=True)
    opt = torch.optim.Adam([B, c, mu, log_sigma], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    for step in range(steps):
        opt.zero_grad()
        eps = torch.randn(n_samples, n_students, k, device=dev)
        theta = (mu + torch.exp(log_sigma) * eps)[:, s, :]  # (S, rows, k)
        logits = torch.einsum("srk,rck->src", theta, B[j]) + c[j][None]
        ll = F.log_softmax(logits, -1).gather(2, o[None, :, None].expand(n_samples, -1, 1))[..., 0].mean(0)
        kl = 0.5 * (mu**2 + torch.exp(2 * log_sigma) - 2 * log_sigma - 1).sum()
        prior = 0.5 * (B**2).sum() / b_sd**2 + 0.5 * (c**2).sum() / 9.0
        loss = -(ll.sum() - kl - prior) / len(option)
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step % 500 == 0 or step == steps - 1):
            print(f"  nominal k={k} step {step:5d}  -elbo/response {loss.item():.4f}")

    return Nominal(B=B.detach().cpu().numpy(), c=c.detach().cpu().numpy(), correct=correct)


def observation_mask(model: Nominal, items: np.ndarray, options: np.ndarray, see_option: bool) -> np.ndarray:
    """Boolean (n, n_options): which options are consistent with what was observed."""
    n_opt = model.c.shape[1]
    if see_option:
        return np.eye(n_opt, dtype=bool)[options]
    is_correct = options == model.correct[items]
    correct_mask = np.eye(n_opt, dtype=bool)[model.correct[items]]
    return np.where(is_correct[:, None], correct_mask, ~correct_mask)


def laplace_nominal(
    B: np.ndarray, c: np.ndarray, M: np.ndarray, theta0: np.ndarray | None = None, iters: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """MAP and covariance of theta under prior N(0, I) given set observations M (n, n_options)."""
    k = B.shape[2]
    theta = np.zeros(k) if theta0 is None else theta0.copy()

    def softmax(logits):
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def moments(theta):
        logits = np.einsum("nck,k->nc", B, theta) + c
        pi = softmax(logits)
        # Renormalize within the observed set in log space; masking pi first can underflow to 0/0.
        piS = softmax(np.where(M, logits, -np.inf))
        m, mS = np.einsum("nc,nck->nk", pi, B), np.einsum("nc,nck->nk", piS, B)
        cov = np.einsum("nc,nck,ncl->kl", pi, B, B) - m.T @ m
        covS = np.einsum("nc,nck,ncl->kl", piS, B, B) - mS.T @ mS
        grad = (mS - m).sum(0) - theta
        return grad, cov - covS + np.eye(k), cov + np.eye(k)

    for _ in range(iters):
        grad, H, H_safe = moments(theta)
        # Set observations make the log-likelihood non-concave; fall back to a PSD curvature if needed.
        if np.linalg.eigvalsh(H).min() <= 1e-6:
            H = H_safe
        step = np.linalg.solve(H, grad)
        norm = np.linalg.norm(step)
        if norm > 1.0:  # damp: full Newton steps can overshoot on the non-concave parts
            step /= norm
        theta += step
        if np.abs(step).max() < 1e-6:
            break
    _, H, H_safe = moments(theta)
    if np.linalg.eigvalsh(H).min() <= 1e-6:
        H = H_safe
    return theta, np.linalg.inv(H)
