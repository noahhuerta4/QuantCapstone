"""Neutralization and the risk pipeline (spec S0: a priority test target).

The failure mode this guards against is the quiet one: a book that reports itself
as market-neutral while carrying a beta tilt that supplies most of its return.
The constraints have to hold *simultaneously* and *exactly*, which is why these
tests check the projection algebra directly rather than just eyeballing the
delivered portfolio.
"""

import numpy as np
import pandas as pd
import pytest

from quant_capstone.config import RiskConfig
from quant_capstone.factors import sector_dummies
from quant_capstone.linalg import build_loadings, project_out
from quant_capstone.risk import (
    cap_gross,
    clip_to_caps,
    enforce,
    neutrality_loadings,
    neutralize,
    per_name_caps,
    portfolio_vol,
    scale_to_vol_target,
)

RNG = np.random.default_rng(11)


def test_projection_removes_the_loading_exactly():
    x = RNG.normal(size=50)
    loadings = np.column_stack([np.ones(50), RNG.normal(size=50)])
    resid = project_out(x, loadings)
    assert np.abs(loadings.T @ resid).max() < 1e-10


def test_projection_is_idempotent():
    x = RNG.normal(size=40)
    loadings = np.column_stack([np.ones(40), RNG.normal(size=40)])
    once = project_out(x, loadings)
    assert project_out(once, loadings) == pytest.approx(once, abs=1e-12)


def test_projection_leaves_an_already_neutral_vector_alone():
    loadings = np.column_stack([np.ones(30), RNG.normal(size=30)])
    neutral = project_out(RNG.normal(size=30), loadings)
    assert project_out(neutral, loadings) == pytest.approx(neutral, abs=1e-12)


def test_projection_survives_rank_deficient_loadings():
    # A duplicated column and an all-zero column are both routine in practice:
    # a one-member sector is collinear with the intercept, an empty sector is zero.
    base = RNG.normal(size=25)
    loadings = np.column_stack([np.ones(25), np.ones(25), np.zeros(25)])
    resid = project_out(base, loadings)
    assert abs(resid.sum()) < 1e-10


def test_projection_ignores_non_finite_entries():
    x = np.array([1.0, 2.0, np.nan, 4.0])
    loadings = np.ones((4, 1))
    resid = project_out(x, loadings)
    assert resid[2] == 0.0
    assert abs(resid[[0, 1, 3]].sum()) < 1e-12


def test_build_loadings_drops_empty_blocks():
    assert build_loadings(None, np.array([])) is None
    assert build_loadings(np.zeros((5, 2))) is None
    assert build_loadings(np.ones((5, 1)), None).shape == (5, 1)


def test_dollar_beta_and_sector_neutrality_hold_together():
    n = 60
    tickers = pd.Index([f"T{i}" for i in range(n)])
    beta = pd.Series(RNG.uniform(0.5, 1.5, n), index=tickers)
    sectors = pd.Series([f"S{i % 5}" for i in range(n)], index=tickers)
    dummies = sector_dummies(sectors, tickers)

    cfg = RiskConfig()
    loadings = neutrality_loadings(tickers, beta, dummies, cfg)
    w = neutralize(RNG.normal(size=n), loadings)

    assert abs(w.sum()) < 1e-10, "dollar neutrality"
    assert abs(w @ beta.to_numpy()) < 1e-10, "beta neutrality"
    net_by_sector = w @ dummies.to_numpy()
    assert np.abs(net_by_sector).max() < 1e-10, "sector neutrality"


def test_disabling_a_constraint_actually_relaxes_it():
    n = 40
    tickers = pd.Index([f"T{i}" for i in range(n)])
    beta = pd.Series(RNG.uniform(0.5, 1.5, n), index=tickers)
    dummies = sector_dummies(pd.Series(["S0"] * n, index=tickers), tickers)

    cfg = RiskConfig(neutralize_beta=False, neutralize_sector=False)
    w = neutralize(RNG.normal(size=n), neutrality_loadings(tickers, beta, dummies, cfg))
    assert abs(w.sum()) < 1e-10
    assert abs(w @ beta.to_numpy()) > 1e-6, "beta should be free when not neutralized"


def test_vol_scaling_hits_the_target_and_preserves_neutrality():
    n = 30
    cov = np.diag(RNG.uniform(1e-4, 4e-4, n))
    loadings = np.ones((n, 1))
    w = project_out(RNG.normal(size=n), loadings)

    scaled = scale_to_vol_target(w, cov, 0.10)
    assert portfolio_vol(scaled, cov) == pytest.approx(0.10, rel=1e-9)
    assert abs(scaled.sum()) < 1e-10, "uniform scaling must not break neutrality"


def test_vol_scaling_of_a_zero_book_is_zero():
    cov = np.eye(5) * 1e-4
    assert scale_to_vol_target(np.zeros(5), cov, 0.1) == pytest.approx(np.zeros(5))


def test_caps_take_the_tighter_of_the_fixed_and_liquidity_limits():
    cfg = RiskConfig(max_weight=0.015, max_adv_participation=0.01)
    # $200m ADV against $10m NAV: liquidity allows 20% of NAV, so the 1.5% cap binds.
    assert per_name_caps(np.array([200e6]), 10e6, cfg)[0] == pytest.approx(0.015)
    # $5m ADV against $1bn NAV: liquidity allows 0.005% and binds hard.
    assert per_name_caps(np.array([5e6]), 1e9, cfg)[0] == pytest.approx(5e-5)


def test_enforce_satisfies_every_cap_without_breaking_neutrality():
    n = 50
    w = project_out(RNG.normal(size=n) * 0.05, np.ones((n, 1)))
    caps = np.full(n, 0.015)

    out = enforce(w, caps, max_gross=2.0)
    assert np.abs(out).max() <= 0.015 + 1e-12
    assert np.abs(out).sum() <= 2.0 + 1e-12
    assert abs(out.sum()) < 1e-10


def test_enforce_zeroes_names_with_no_liquidity():
    w = np.array([0.01, 0.01, -0.02])
    caps = np.array([0.015, 0.0, 0.015])
    out = enforce(w, caps, max_gross=5.0)
    assert out[1] == 0.0
    assert out[0] != 0.0, "a zero cap on one name must not wipe out the book"


def test_clip_and_gross_cap():
    w = np.array([0.05, -0.05, 0.02])
    assert np.abs(clip_to_caps(w, np.full(3, 0.03))).max() == pytest.approx(0.03)
    capped = cap_gross(np.array([1.0, -1.0]), 1.0)
    assert np.abs(capped).sum() == pytest.approx(1.0)
    assert capped[0] == pytest.approx(-capped[1]), "uniform scaling keeps the book neutral"


def test_delivered_portfolio_is_neutral_and_respects_caps(run, cfg):
    weights = run.portfolio.weights
    live = weights.loc[weights.abs().sum(axis=1) > 0]
    assert len(live) > 100

    assert live.sum(axis=1).abs().max() < 1e-8, "dollar neutral"
    assert live.abs().max().max() <= cfg.risk.max_weight + 1e-9, "per-name cap"
    assert live.abs().sum(axis=1).max() <= cfg.risk.max_gross + 1e-9, "gross cap"

    beta = run.portfolio.beta.reindex_like(live).fillna(1.0)
    assert (live * beta).sum(axis=1).abs().max() < 1e-6, "beta neutral"


def test_realized_beta_of_the_book_is_near_zero(run):
    assert abs(run.realized_beta()) < 0.1
