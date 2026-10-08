"""Corwin-Schultz spread estimation and per-name slippage (spec S6).

The estimator is the input that decides whether illiquid names look expensive to
trade, so these tests check it recovers a *known* spread rather than merely
returning finite numbers.
"""

import numpy as np
import pandas as pd
import pytest

from quant_capstone.config import CostConfig
from quant_capstone.costs import (
    CostModel,
    SlippageModel,
    corwin_schultz_spread,
    rolling_half_spread_bps,
    spread_summary,
)


def _bars_with_spread(spread: float, n_days: int = 400, seed: int = 0, daily_vol: float = 0.015):
    """Synthetic OHLC where the true proportional spread is known."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n_days)
    mid = 100.0 * np.exp(np.cumsum(rng.normal(0, daily_vol, n_days)))
    # Intraday range from a Brownian path, then widened by the bid-ask bounce.
    swing = np.abs(rng.normal(0, daily_vol, n_days)) + daily_vol
    high = mid * (1 + swing) * (1 + spread / 2)
    low = mid * (1 - swing) * (1 - spread / 2)
    to_frame = lambda a: pd.DataFrame({"A": a}, index=index)
    return to_frame(high), to_frame(low)


def test_estimator_recovers_a_wider_spread_as_wider():
    tight_h, tight_l = _bars_with_spread(0.001)
    wide_h, wide_l = _bars_with_spread(0.02)

    tight = corwin_schultz_spread(tight_h, tight_l)["A"].median()
    wide = corwin_schultz_spread(wide_h, wide_l)["A"].median()
    assert wide > tight, "a 200 bps spread must estimate above a 10 bps one"


def test_estimate_is_monotone_in_the_true_spread():
    medians = [
        corwin_schultz_spread(*_bars_with_spread(s))["A"].median()
        for s in (0.001, 0.005, 0.01, 0.02, 0.04)
    ]
    assert all(a < b for a, b in zip(medians, medians[1:]))


def test_spread_is_never_negative():
    high, low = _bars_with_spread(0.0)
    assert (corwin_schultz_spread(high, low).fillna(0.0) >= 0).all().all()


def test_rolling_estimate_is_point_in_time():
    """Row t reads t+1, so the usable series must lag by two days."""
    high, low = _bars_with_spread(0.01, n_days=300)
    rolling = rolling_half_spread_bps(high, low, window=21, min_periods=5)

    original = rolling.copy()
    high2, low2 = high.copy(), low.copy()
    high2.iloc[200:] *= 3.0
    low2.iloc[200:] /= 3.0
    tampered = rolling_half_spread_bps(high2, low2, window=21, min_periods=5)

    # Blowing up the future must not change any estimate at or before day 200.
    pd.testing.assert_frame_equal(original.iloc[:201], tampered.iloc[:201])


def test_rolling_estimate_is_reported_in_basis_points():
    high, low = _bars_with_spread(0.01, n_days=300)
    half = rolling_half_spread_bps(high, low).stack().median()
    # True half-spread is 50 bps; the estimator is noisy but must land in range.
    assert 10 < half < 200


def test_spread_summary_reports_a_distribution():
    high, low = _bars_with_spread(0.01, n_days=300)
    stats = spread_summary(rolling_half_spread_bps(high, low))
    assert stats["p05_bps"] <= stats["median_bps"] <= stats["p95_bps"]


# --------------------------------------------------------- per-name slippage
def test_per_name_spread_overrides_the_flat_rate():
    model = SlippageModel(half_spread_bps=2.0, impact_coef=0.0, use_impact=False)
    rate = model.rate(
        np.array([100.0, 100.0]), np.array([1e6, 1e6]), np.array([0.02, 0.02]),
        half_spread_bps=np.array([5.0, 40.0]),
    )
    assert rate == pytest.approx([5e-4, 40e-4])


def test_missing_per_name_estimates_fall_back_to_the_configured_rate():
    model = SlippageModel(half_spread_bps=2.0, impact_coef=0.0, use_impact=False)
    rate = model.rate(
        np.array([100.0, 100.0]), np.array([1e6, 1e6]), np.array([0.02, 0.02]),
        half_spread_bps=np.array([np.nan, 30.0]),
    )
    assert rate == pytest.approx([2e-4, 30e-4])


def test_cost_model_charges_more_for_a_wider_spread():
    kwargs = dict(
        trade_shares=np.array([1000.0]),
        trade_notional=np.array([50_000.0]),
        position_notional=np.array([50_000.0]),
        adv_shares=np.array([1e6]),
        sigma=np.array([0.02]),
    )
    model = CostModel(CostConfig())
    tight = model.charge(**kwargs, half_spread_bps=np.array([2.0]))
    wide = model.charge(**kwargs, half_spread_bps=np.array([40.0]))

    assert wide.slippage > tight.slippage
    assert wide.commission == pytest.approx(tight.commission), "only slippage should move"


def test_estimated_spread_flag_changes_the_backtest(dataset, cfg, run):
    from quant_capstone.backtest import run_backtest

    flat = run_backtest(
        dataset, run.portfolio.weights, cfg.override(costs__use_estimated_spread=False)
    )
    estimated = run_backtest(
        dataset, run.portfolio.weights, cfg.override(costs__use_estimated_spread=True)
    )
    assert not np.isclose(flat.costs["slippage"].sum(), estimated.costs["slippage"].sum())


def test_liquidity_scaled_spread_is_decreasing_and_bounded():
    from quant_capstone.costs import liquidity_scaled_half_spread
    from quant_capstone.costs.spread import SPREAD_CAP_BPS, SPREAD_FLOOR_BPS

    adv = pd.DataFrame({"a": [1e9], "b": [1e8], "c": [1e7], "d": [1e6]})
    bps = liquidity_scaled_half_spread(adv).iloc[0]

    assert bps["a"] < bps["b"] < bps["c"] < bps["d"], "cheaper to trade when more liquid"
    assert bps.between(SPREAD_FLOOR_BPS, SPREAD_CAP_BPS).all()
    # Calibration anchors: ~0.2-0.5 bps for a mega-cap, ~10 bps at $10m ADV.
    assert bps["a"] < 1.0
    assert 5.0 < bps["c"] < 20.0
