"""Small helpers every study uses: run one backtest, summarize a sleeve, load S&P 1500."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest import run_backtest
from ..config import TRADING_DAYS, Config, DataConfig
from ..data import YFinanceSource, clean_prices
from ..data.base import PriceData
from ..data.loader import Dataset
from ..data.universe import build_universe
from ..law import ic_summary, information_coefficient
from ..reporting.metrics import sharpe
from ..risk.pipeline import build_portfolio
from ..signals import build_signal

MIN_DENSITY = 0.8
"""A rate-limited download comes back structurally complete and nearly all null.
Every study refuses to run on a panel sparser than this (see data/yfinance_source)."""


def assert_dense(prices: PriceData) -> None:
    density = float(prices.close.notna().to_numpy().mean())
    if density <= MIN_DENSITY:
        raise RuntimeError(f"price panel density {density:.2f} -- cache suspect, refusing to run")


def backtest(dataset: Dataset, cfg: Config, weights: pd.DataFrame | None = None):
    """Signal -> portfolio -> backtest for one config, or backtest given weights."""
    if weights is None:
        weights = build_portfolio(dataset, build_signal(dataset, cfg), cfg).weights
    return run_backtest(dataset, weights, cfg)


def cost_bps(result) -> float:
    return float(result.costs.sum(axis=1).div(result.nav).mean() * TRADING_DAYS * 1e4)


def turnover(result) -> float:
    return float(result.turnover.mean() * TRADING_DAYS)


def summary(result) -> dict[str, float]:
    """The five numbers most studies tabulate per backtest."""
    return {
        "gross": sharpe(result.gross_return),
        "net": sharpe(result.net_return),
        "turnover": turnover(result),
        "cost_bps": cost_bps(result),
        "trades_day": float(result.n_trades.mean()),
    }


def autocorr(sleeve: pd.DataFrame) -> float:
    """Mean day-over-day cross-sectional correlation -- the breadth killer."""
    return float(sleeve.corrwith(sleeve.shift(1), axis=1).mean())


def sleeve_row(name: str, sleeve: pd.DataFrame, fwd: pd.DataFrame,
               members: pd.DataFrame | None = None) -> dict[str, float]:
    stats = ic_summary(information_coefficient(sleeve, fwd).dropna())
    row = {"sleeve": name}
    if members is not None:
        row["coverage"] = float(sleeve.notna().sum().sum() / members.sum().sum())
    row.update({"ic_mean": stats["ic_mean"], "ic_t": stats["ic_t_stat"], "autocorr": autocorr(sleeve)})
    return row


def fixed_blend(base: Config, weights: dict[str, float]) -> Config:
    """A config that blends exactly these sleeves at exactly these weights."""
    return base.override(
        signal__alphas=tuple(weights), signal__pead_weight=0.0,
        signal__combine="fixed", signal__sleeve_weights=tuple(weights.items()),
    )


def sp1500_prices(min_observations: int) -> PriceData:
    """The wider S&P 1500 panel the liquidity and breadth studies need."""
    source = YFinanceSource(cache_dir="data_cache")
    tickers = source.sp1500_constituents()["ticker"].tolist()
    prices = clean_prices(source.fetch(tickers, "2015-01-01", "2025-01-01"),
                          min_observations=min_observations)
    assert_dense(prices)
    return prices


def sub_universe(prices: PriceData, universe_size: int, min_history: int) -> Dataset:
    """A dataset over ``prices`` restricted to the top ``universe_size`` names by ADV."""
    dcfg = DataConfig(universe_size=universe_size, adv_lookback=63,
                      min_history=min_history, min_price=5.0)
    return Dataset(prices, build_universe(prices, dcfg), dcfg)


def t_stat(values: np.ndarray | pd.Series, ddof: int = 0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean() / values.std(ddof=ddof) * np.sqrt(len(values)))
