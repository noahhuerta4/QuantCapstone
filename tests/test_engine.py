"""Backtest engine accounting.

If these pass, the P&L, NAV and turnover numbers are internally consistent and
match an independently written expression for the same quantity.
"""

import numpy as np
import pandas as pd
import pytest

from quant_capstone.backtest import analytic_gross_returns, replication_check, run_backtest
from quant_capstone.backtest.crosscheck import lookahead_probe


def test_engine_matches_the_analytic_expression(dataset, run, cfg):
    check = replication_check(dataset, run.portfolio.weights, cfg)
    assert check["passes"] == 1.0
    assert check["max_abs_difference"] < 1e-9
    assert check["correlation"] == pytest.approx(1.0)


def test_nav_compounds_at_the_net_return(run):
    result = run.result
    expected = result.nav.shift(1) * (1.0 + result.net_return.shift(1))
    diff = (result.nav - expected).abs().iloc[1:]
    assert diff.max() < 1e-6


def test_net_is_gross_minus_costs(run):
    result = run.result
    implied = result.costs.sum(axis=1) / result.nav
    assert (result.cost_return - implied).abs().max() < 1e-12


def test_costs_are_never_negative(run):
    assert (run.result.costs >= 0).all().all()
    assert (run.result.net_return <= run.result.gross_return + 1e-15).all()


def test_a_zero_signal_produces_a_flat_book(dataset, cfg):
    weights = pd.DataFrame(
        0.0, index=dataset.prices.dates, columns=dataset.prices.tickers
    )
    result = run_backtest(dataset, weights, cfg)
    assert result.gross_return.abs().max() == 0.0
    assert result.turnover.max() == 0.0
    assert result.nav.nunique() == 1


def test_no_trade_band_reduces_turnover(dataset, run, cfg):
    weights = run.portfolio.weights
    tight = run_backtest(dataset, weights, cfg.override(execution__no_trade_band=0.0))
    loose = run_backtest(dataset, weights, cfg.override(execution__no_trade_band=0.01))

    assert loose.turnover.mean() < tight.turnover.mean()
    assert loose.n_trades.mean() < tight.n_trades.mean()


def test_rebalance_frequency_reduces_trading_days(dataset, run, cfg):
    weights = run.portfolio.weights
    every_third = run_backtest(dataset, weights, cfg.override(execution__rebalance_every=3))
    daily = run_backtest(dataset, weights, cfg.override(execution__rebalance_every=1))
    assert (every_third.n_trades > 0).sum() < (daily.n_trades > 0).sum()


def test_turnover_equals_traded_notional_over_nav(run):
    result = run.result
    implied = result.trades.abs().sum(axis=1) / result.nav
    assert (result.turnover - implied).abs().max() < 1e-12


def test_exposures_are_consistent_with_the_weights(run):
    weights, exposures = run.result.weights, run.result.exposures
    assert (exposures["gross"] - weights.abs().sum(axis=1)).abs().max() < 1e-12
    assert (exposures["long"] + exposures["short"] - exposures["net"]).abs().max() < 1e-12


def test_execution_timing_changes_the_result(dataset, run, cfg):
    weights = run.portfolio.weights
    next_open = run_backtest(dataset, weights, cfg.override(execution__timing="next_open"))
    same_close = run_backtest(dataset, weights, cfg.override(execution__timing="same_close"))
    assert not np.allclose(next_open.net_return, same_close.net_return)


def test_unknown_timing_is_rejected(dataset, run, cfg):
    with pytest.raises(ValueError):
        run_backtest(dataset, run.portfolio.weights, cfg.override(execution__timing="teleport"))


def test_engine_is_sensitive_to_the_timing_it_claims(dataset, run, cfg):
    """A backtest that scores the same on shifted weights is not using them."""
    probe = lookahead_probe(dataset, run.portfolio.weights, cfg)
    assert probe["passes"] == 1.0


def test_analytic_returns_ignore_weights_outside_the_panel(dataset, run, cfg):
    weights = run.portfolio.weights.copy()
    weights["NOT_A_REAL_TICKER"] = 1.0
    series = analytic_gross_returns(dataset, weights, cfg)
    assert np.isfinite(series).all()
