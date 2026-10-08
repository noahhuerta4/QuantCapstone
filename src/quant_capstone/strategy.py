"""End-to-end wiring: config -> data -> signal -> portfolio -> backtest.

One function so the validation harness, the CLI and the notebooks all run exactly
the same path. If you find yourself reassembling these stages by hand somewhere
else, that is a bug waiting to happen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .backtest.engine import run_backtest
from .backtest.results import BacktestResult
from .config import Config
from .data.loader import Dataset, load_dataset
from .factors import market_return, realized_beta
from .risk.pipeline import PortfolioOutput, build_portfolio
from .signals.build import build_signal

log = logging.getLogger(__name__)


@dataclass
class StrategyRun:
    cfg: Config
    dataset: Dataset
    signal: pd.DataFrame
    portfolio: PortfolioOutput
    result: BacktestResult

    @property
    def gross(self) -> pd.Series:
        return self.result.gross_return

    @property
    def net(self) -> pd.Series:
        return self.result.net_return

    def realized_beta(self) -> float:
        mkt = market_return(self.dataset.returns, self.dataset.members)
        return realized_beta(self.net, mkt)


def run_strategy(cfg: Config, dataset: Dataset | None = None) -> StrategyRun:
    """Run the whole pipeline. Pass ``dataset`` to reuse a loaded panel.

    A reused panel gets its universe rebuilt if ``cfg.data`` has changed, so that
    sweeping a universe parameter against a cached dataset actually varies the
    universe instead of silently reusing the old one.
    """
    dataset = dataset.rebuilt_for(cfg.data) if dataset is not None else load_dataset(cfg)
    signal = build_signal(dataset, cfg)
    portfolio = build_portfolio(dataset, signal, cfg)
    result = run_backtest(dataset, portfolio.weights, cfg)
    return StrategyRun(cfg=cfg, dataset=dataset, signal=signal, portfolio=portfolio, result=result)


def performance_table(result: BacktestResult, label: str = "full") -> pd.DataFrame:
    """Gross and net side by side -- never one without the other (spec S0)."""
    from .reporting.metrics import summarize

    table = pd.concat(
        [summarize(result.gross_return, "gross"), summarize(result.net_return, "net")],
        axis=1,
    )
    table.columns = pd.MultiIndex.from_product([[label], table.columns])
    return table


def activity_table(result: BacktestResult) -> pd.Series:
    """Turnover and trade-count diagnostics for the Section 8 band check."""
    from .config import TRADING_DAYS

    active = result.n_trades[result.n_trades > 0]
    return pd.Series(
        {
            "mean_trades_per_day": float(result.n_trades.mean()),
            # With a multi-day rebalance the calendar mean is diluted by the days
            # we do not trade at all, which can put it inside the target band
            # while every actual trading day sits far above it. Report both.
            "mean_trades_per_trading_day": float(active.mean()) if len(active) else 0.0,
            "median_trades_per_day": float(result.n_trades.median()),
            "p95_trades_per_day": float(result.n_trades.quantile(0.95)),
            "days_with_trades": float(len(active)),
            "fraction_of_days_trading": float(len(active) / max(len(result.n_trades), 1)),
            "mean_daily_turnover": float(result.turnover.mean()),
            "annualized_turnover": float(result.turnover.mean() * TRADING_DAYS),
            "mean_gross_exposure": float(result.exposures["gross"].mean()),
            "mean_net_exposure": float(result.exposures["net"].mean()),
            "mean_adv_participation": float(result.participation[result.participation > 0].mean()),
        }
    )
