"""Point-in-time discipline (spec S0).

The first test here is load-bearing for the whole validation harness: it proves
the pipeline is **causal**, which is the assumption that lets ``validation/grid.py``
run each configuration once over the full history and slice it per fold instead
of re-running it. If this test ever fails, every walk-forward number in the
project silently becomes in-sample.
"""

import numpy as np
import pandas as pd
import pytest

from quant_capstone.data.loader import Dataset
from quant_capstone.data.universe import build_universe
from quant_capstone.reporting.metrics import sharpe
from quant_capstone.risk.pipeline import build_portfolio
from quant_capstone.signals import build_signal
from quant_capstone.strategy import run_strategy


def test_weights_do_not_change_when_the_future_is_removed(dataset, cfg):
    """Truncating the panel must leave every earlier weight bit-identical."""
    full = build_portfolio(dataset, build_signal(dataset, cfg), cfg)

    cut = dataset.prices.dates[900]
    truncated_prices = dataset.prices.slice_dates(end=cut)
    truncated = Dataset(truncated_prices, build_universe(truncated_prices, cfg.data))
    partial = build_portfolio(truncated, build_signal(truncated, cfg), cfg)

    overlap = partial.weights.index
    pd.testing.assert_frame_equal(
        full.weights.loc[overlap], partial.weights, check_exact=False, atol=1e-12
    )


def test_universe_membership_is_unchanged_by_the_future(dataset, cfg):
    cut = dataset.prices.dates[700]
    truncated = build_universe(dataset.prices.slice_dates(end=cut), cfg.data)
    pd.testing.assert_frame_equal(dataset.members.loc[truncated.index], truncated)


def test_a_random_signal_earns_nothing(dataset, cfg):
    """Sanity floor: the machinery must not manufacture a Sharpe from noise."""
    rng = np.random.default_rng(3)
    noise = pd.DataFrame(
        rng.normal(size=dataset.members.shape),
        index=dataset.members.index,
        columns=dataset.members.columns,
    ).where(dataset.members)

    from quant_capstone.backtest import run_backtest

    portfolio = build_portfolio(dataset, noise, cfg)
    result = run_backtest(dataset, portfolio.weights, cfg)
    assert abs(sharpe(result.gross_return)) < 1.5
    assert sharpe(result.net_return) < sharpe(result.gross_return)


def test_reversing_the_signal_reverses_the_edge(dataset, cfg):
    from quant_capstone.backtest import run_backtest

    signal = build_signal(dataset, cfg)
    forward = run_backtest(dataset, build_portfolio(dataset, signal, cfg).weights, cfg)
    backward = run_backtest(dataset, build_portfolio(dataset, -signal, cfg).weights, cfg)

    assert sharpe(forward.gross_return) > 0
    assert sharpe(backward.gross_return) < 0


def test_shuffling_names_within_each_day_destroys_the_edge(dataset, cfg):
    """Guards against a bug that pays off on cross-sectional shape alone."""
    from quant_capstone.backtest import run_backtest

    rng = np.random.default_rng(5)
    signal = build_signal(dataset, cfg)
    values = signal.to_numpy(copy=True)
    for row in values:
        present = np.flatnonzero(np.isfinite(row))
        row[present] = row[rng.permutation(present)]
    shuffled = pd.DataFrame(values, index=signal.index, columns=signal.columns)

    portfolio = build_portfolio(dataset, shuffled, cfg)
    result = run_backtest(dataset, portfolio.weights, cfg)
    real = run_strategy(cfg, dataset)
    assert sharpe(result.gross_return) < sharpe(real.result.gross_return) / 2


def test_signal_never_scores_a_name_outside_the_universe(signal, dataset):
    outside = signal.notna() & ~dataset.members
    assert not outside.to_numpy().any()
