"""Performance metrics (spec S11)."""

import numpy as np
import pandas as pd
import pytest

from quant_capstone.config import TRADING_DAYS
from quant_capstone.reporting.metrics import (
    annualized_return,
    annualized_volatility,
    calmar,
    conditional_var,
    drawdown_duration,
    drawdown_series,
    hit_rate,
    max_drawdown,
    rolling_sharpe,
    sharpe,
    sortino,
    summarize,
    value_at_risk,
)


def _series(values) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_sharpe_matches_the_definition():
    rng = np.random.default_rng(0)
    r = _series(rng.normal(0.0005, 0.01, 1000))
    expected = r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    assert sharpe(r) == pytest.approx(expected)


def test_sharpe_of_a_constant_series_is_undefined():
    assert np.isnan(sharpe(_series([0.001] * 100)))


def test_sharpe_is_scale_invariant():
    rng = np.random.default_rng(1)
    r = _series(rng.normal(0.001, 0.01, 500))
    assert sharpe(r * 3) == pytest.approx(sharpe(r))


def test_sortino_exceeds_sharpe_for_positively_skewed_returns():
    r = _series([0.02] * 60 + [-0.005] * 40)
    assert sortino(r) > sharpe(r)


def test_annualized_return_is_geometric():
    # 100 days of exactly +0.1% compounds to 1.001**100.
    r = _series([0.001] * 100)
    expected = (1.001**100) ** (TRADING_DAYS / 100) - 1
    assert annualized_return(r) == pytest.approx(expected)


def test_annualized_volatility_scales_with_the_square_root_of_time():
    rng = np.random.default_rng(2)
    r = _series(rng.normal(0, 0.01, 2000))
    assert annualized_volatility(r) == pytest.approx(0.01 * np.sqrt(TRADING_DAYS), rel=0.1)


def test_max_drawdown_finds_the_worst_peak_to_trough():
    # +10%, then -50% off the peak, then partial recovery.
    r = _series([0.10, -0.50, 0.20])
    assert max_drawdown(r) == pytest.approx(-0.50)
    assert drawdown_series(r).iloc[1] == pytest.approx(-0.50)


def test_no_drawdown_for_a_monotonically_rising_curve():
    assert max_drawdown(_series([0.01] * 50)) == pytest.approx(0.0)


def test_drawdown_duration_measures_time_underwater():
    # Down for three days, recovering exactly on the fourth.
    r = _series([-0.1, -0.1, 0.05, 0.0, 0.0, 0.5, 0.01])
    stats = drawdown_duration(r)
    assert stats["max_drawdown"] < 0
    assert stats["longest_underwater_days"] >= 3
    assert stats["max_drawdown_days"] > 0


def test_calmar_is_return_over_drawdown():
    r = _series([0.01] * 50 + [-0.02] * 5 + [0.01] * 50)
    assert calmar(r) == pytest.approx(annualized_return(r) / abs(max_drawdown(r)))


def test_hit_rate_ignores_flat_days():
    assert hit_rate(_series([0.01, -0.01, 0.0, 0.01])) == pytest.approx(2 / 3)


def test_var_is_reported_as_a_positive_loss():
    rng = np.random.default_rng(3)
    r = _series(rng.normal(0, 0.01, 5000))
    var = value_at_risk(r, 0.05)
    assert var > 0
    assert var == pytest.approx(0.01 * 1.645, rel=0.1)


def test_cvar_is_worse_than_var():
    rng = np.random.default_rng(4)
    r = _series(rng.normal(0, 0.01, 5000))
    assert conditional_var(r, 0.05) > value_at_risk(r, 0.05)


def test_parametric_and_historical_var_agree_for_normal_returns():
    rng = np.random.default_rng(5)
    r = _series(rng.normal(0, 0.01, 20000))
    assert value_at_risk(r, 0.05, "historical") == pytest.approx(
        value_at_risk(r, 0.05, "parametric"), rel=0.05
    )


def test_unknown_var_method_is_rejected():
    with pytest.raises(ValueError):
        value_at_risk(_series([0.01, -0.01]), 0.05, "vibes")


def test_rolling_sharpe_has_the_requested_window():
    rng = np.random.default_rng(6)
    r = _series(rng.normal(0.0005, 0.01, 500))
    rs = rolling_sharpe(r, 126)
    assert rs.iloc[:62].isna().all()
    assert rs.dropna().notna().all()


def test_summarize_returns_the_full_block():
    rng = np.random.default_rng(7)
    stats = summarize(_series(rng.normal(0.0005, 0.01, 800)), "net")
    for key in ("sharpe", "sortino", "calmar", "max_drawdown", "cvar_95", "n_days"):
        assert key in stats.index
    assert stats["n_days"] == 800


def test_metrics_tolerate_gaps_and_infinities():
    r = _series([0.01, np.nan, -0.01, np.inf, 0.02, 0.0])
    assert np.isfinite(sharpe(r))
    assert np.isfinite(max_drawdown(r))
