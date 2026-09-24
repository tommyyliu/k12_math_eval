import numpy as np
import pandas as pd

from k12diag.data import densest_window
from k12diag.irt import IRT2PL, fit_2pl, make_grid
from k12diag.policies import ExpectedLossReductionPolicy, _entropy


def test_fit_2pl_recovers_parameters():
    rng = np.random.default_rng(0)
    n_students, n_items = 2000, 30
    theta = rng.normal(size=n_students)
    a = rng.uniform(0.5, 2.0, n_items)
    b = rng.normal(size=n_items)
    s, j = np.meshgrid(np.arange(n_students), np.arange(n_items), indexing="ij")
    s, j = s.ravel(), j.ravel()
    y = (rng.random(len(s)) < 1 / (1 + np.exp(-a[j] * (theta[s] - b[j])))).astype(float)

    m = fit_2pl(s, j, y, n_students, n_items, steps=400)
    assert np.corrcoef(m.b, b)[0, 1] > 0.97
    assert np.corrcoef(m.a, a)[0, 1] > 0.85


def test_elr_matches_brute_force():
    rng = np.random.default_rng(1)
    theta, log_prior = make_grid(41)
    m = IRT2PL(a=rng.uniform(0.5, 2, 12), b=rng.normal(size=12), theta=theta, log_prior=log_prior)
    P = m.p_correct_grid()
    w = np.exp(log_prior + rng.normal(scale=0.3, size=len(theta)))
    w /= w.sum()
    candidates = np.arange(5)
    scores = ExpectedLossReductionPolicy().scores(w, P, candidates)

    for k, q in enumerate(candidates):
        before = _entropy(w @ P).sum()
        after = 0.0
        for lik in (P[:, q], 1 - P[:, q]):
            p_y = w @ lik
            post = w * lik / p_y
            after += p_y * _entropy(post @ P).sum()
        assert np.isclose(scores[k], before - after)
    assert (scores > 0).all()


def test_densest_window_picks_busiest_period():
    dates = pd.Series(
        pd.to_datetime(["2020-01-01", "2020-03-01", "2020-03-02", "2020-03-05", "2020-06-01"])
    )
    mask = densest_window(dates, days=10)
    assert mask.tolist() == [False, True, True, True, False]


def test_mirt_prefers_true_dimensionality():
    import torch

    from k12diag.mirt import fit_mirt, infer_students, predict

    rng = np.random.default_rng(2)
    n_students, n_items = 1500, 40
    theta = rng.normal(size=(n_students, 2))
    A = np.zeros((n_items, 2))
    A[:20, 0] = rng.uniform(1, 2, 20)  # two unrelated skills
    A[20:, 1] = rng.uniform(1, 2, 20)
    d = rng.normal(size=n_items)
    y = (rng.random((n_students, n_items)) < 1 / (1 + np.exp(-(theta @ A.T - d)))).astype(np.float32)
    s, j = np.meshgrid(np.arange(n_students), np.arange(n_items), indexing="ij")
    train = s < 1200
    obs = (s >= 1200) & (j % 2 == 0)
    tgt = (s >= 1200) & (j % 2 == 1)

    losses = {}
    for k in (1, 2):
        m = fit_mirt(s[train], j[train], y[train], n_students, n_items, k, steps=800,
                     device=torch.device("cpu"))
        mu, sig = infer_students(m, s[obs] - 1200, j[obs], y[obs], 300, device=torch.device("cpu"))
        p = predict(m, mu, sig, s[tgt] - 1200, j[tgt])
        losses[k] = -np.mean(y[tgt] * np.log(p) + (1 - y[tgt]) * np.log(1 - p))
    assert losses[2] < losses[1] - 0.02


def test_mirt_elr_matches_grid_elr():
    """With uniform sample weights, the torch ELR equals the numpy grid ELR on the same matrix."""
    import torch

    from k12diag.mirt_cat import MExpectedLossReduction

    rng = np.random.default_rng(3)
    Ps = rng.uniform(0.05, 0.95, size=(200, 30))
    ref = np.arange(30)
    cand = np.arange(10)
    w = np.full(200, 1 / 200)
    expected = ExpectedLossReductionPolicy(ref).scores(w, Ps, cand)
    got = MExpectedLossReduction(ref, device=torch.device("cpu")).scores(Ps, cand)
    assert np.allclose(got, expected, rtol=1e-3, atol=1e-3)


def test_laplace_posterior_recovers_ability():
    from k12diag.mirt_cat import laplace_posterior

    rng = np.random.default_rng(4)
    theta = np.array([1.0, -0.5])
    A = rng.uniform(0.5, 2.0, size=(400, 2)) * rng.integers(0, 2, size=(400, 2))
    d = rng.normal(size=400)
    y = (rng.random(400) < 1 / (1 + np.exp(-(A @ theta - d)))).astype(float)
    mu, cov = laplace_posterior(A, d, y)
    assert np.all(np.abs(mu - theta) < 3 * np.sqrt(np.diag(cov)) + 0.1)
    assert np.all(np.diag(cov) < 0.05)


def _nominal_toy(rng, k=2, n_items=300):
    from k12diag.nominal import Nominal

    B = rng.normal(size=(n_items, 4, k))
    c = rng.normal(size=(n_items, 4))
    return Nominal(B=B, c=c, correct=rng.integers(0, 4, n_items))


def test_nominal_laplace_recovers_ability_and_options_help():
    from k12diag.nominal import laplace_nominal, observation_mask

    rng = np.random.default_rng(5)
    m = _nominal_toy(rng)
    theta = np.array([0.8, -1.0])
    items = np.arange(300)
    P = m.option_probs(theta[None], items)[0]
    options = np.array([rng.choice(4, p=p) for p in P])

    covs = {}
    for see in (True, False):
        M = observation_mask(m, items, options, see_option=see)
        mu, cov = laplace_nominal(m.B, m.c, M)
        assert np.all(np.abs(mu - theta) < 4 * np.sqrt(np.diag(cov)) + 0.1)
        covs[see] = cov
    # Seeing the chosen option should be more informative than seeing only correctness.
    assert np.linalg.det(covs[True]) < np.linalg.det(covs[False])


def test_observation_mask():
    from k12diag.nominal import Nominal, observation_mask

    m = Nominal(B=np.zeros((2, 4, 1)), c=np.zeros((2, 4)), correct=np.array([1, 3]))
    M = observation_mask(m, np.array([0, 1]), np.array([1, 0]), see_option=False)
    assert M.tolist() == [[False, True, False, False], [True, True, True, False]]
    M = observation_mask(m, np.array([0, 1]), np.array([1, 0]), see_option=True)
    assert M.tolist() == [[False, True, False, False], [True, False, False, False]]
