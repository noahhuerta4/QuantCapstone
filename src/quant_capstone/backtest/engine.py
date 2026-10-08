"""Vectorized cross-sectional backtester (spec S9).

Timing, stated once and enforced here rather than scattered through the code:

  ``next_open`` (default)
      Signal formed at the day-t close -> weights ``w[t]`` -> traded at the
      **t+1 open** -> held to the t+2 open. So on engine day d we execute
      ``w[d-1]`` at ``open_raw[d]`` and earn ``open[d+1]/open[d] - 1``, booked to
      day d. Nothing the engine touches on day d postdates the d open except the
      return it is about to earn, which is the return of a position already set.

  ``same_close``
      Execute ``w[d]`` at ``close_raw[d]``, earn ``close[d+1]/close[d] - 1``.
      Optimistic — it assumes you can trade at the same close that produced the
      signal — and exists only as a comparison case. Say so in any result.

The loop is over days, vectorized across the cross-section. It is not a
for-loop-over-everything: NAV compounding, dollar-denominated per-order
commission minimums, and position drift between rebalances all make the day step
genuinely sequential, and pretending otherwise is how cost models end up wrong.
2500 days x 200 names runs in about a second.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config
from ..costs.model import COMPONENTS, CostModel
from ..data.loader import Dataset

log = logging.getLogger(__name__)

SIGMA_LOOKBACK = 63


def _execution_frames(prices, timing: str):
    """(execution price, holding return) for the configured timing convention."""
    if timing == "next_open":
        return prices.open_raw, prices.open.shift(-1) / prices.open - 1.0
    if timing == "same_close":
        return prices.close_raw, prices.close.shift(-1) / prices.close - 1.0
    raise ValueError(f"unknown execution timing {timing!r}")


def forward_returns(prices, timing: str) -> pd.DataFrame:
    """Return earned by acting on the signal formed on day t, indexed by t.

    This is the only correct target for an IC calculation: measuring the signal
    against the plain next-day close-to-close return would score a forecast the
    strategy never gets to trade. Under ``next_open`` the position formed from
    signal t is not established until the t+1 open, so the return that matters is
    open[t+2]/open[t+1] - 1 -- one day further out than intuition suggests.
    """
    _, hold = _execution_frames(prices, timing)
    return hold.shift(-1) if timing == "next_open" else hold


def run_backtest(dataset: Dataset, weights: pd.DataFrame, cfg: Config) -> "BacktestResult":
    from .results import BacktestResult

    prices = dataset.prices
    ex = cfg.execution
    dates, tickers = prices.dates, prices.tickers
    n_days, n_assets = len(dates), len(tickers)

    exec_price_df, hold_ret_df = _execution_frames(prices, ex.timing)
    # For next_open we execute the *previous* day's decision; same_close trades
    # on the same row. This single shift is the entire look-ahead control.
    targets_df = weights.shift(1) if ex.timing == "next_open" else weights
    targets_df = targets_df.reindex(index=dates, columns=tickers)

    sigma_df = prices.returns().rolling(SIGMA_LOOKBACK, min_periods=20).std().shift(1)
    adv_shares_df = prices.adv_shares(cfg.data.adv_lookback)

    spread_df = None
    if cfg.costs.use_estimated_spread:
        from ..costs.spread import liquidity_scaled_half_spread, rolling_half_spread_bps

        if cfg.costs.spread_estimator == "liquidity_scaled":
            spread_df = liquidity_scaled_half_spread(prices.adv(cfg.data.adv_lookback))
        else:
            spread_df = rolling_half_spread_bps(
                prices.high, prices.low, prices.close,
                window=cfg.costs.spread_window, estimator=cfg.costs.spread_estimator,
            )

    exec_price = exec_price_df.to_numpy(dtype=float)
    hold_ret = np.nan_to_num(hold_ret_df.to_numpy(dtype=float), nan=0.0)
    targets = np.nan_to_num(targets_df.to_numpy(dtype=float), nan=0.0)
    sigma = np.nan_to_num(sigma_df.to_numpy(dtype=float), nan=0.0)
    adv_shares = np.nan_to_num(adv_shares_df.to_numpy(dtype=float), nan=0.0)
    # NaN is meaningful here: the slippage model falls back to the configured
    # flat spread wherever the per-name estimate is unavailable.
    spread = (
        spread_df.reindex(index=dates, columns=tickers).to_numpy(dtype=float)
        if spread_df is not None
        else None
    )

    cost_model = CostModel(cfg.costs)

    nav = float(ex.initial_nav)
    pos = np.zeros(n_assets)  # signed dollar positions, drifted to the current open

    out_nav = np.empty(n_days)
    out_gross = np.zeros(n_days)
    out_net = np.zeros(n_days)
    out_costs = np.zeros((n_days, len(COMPONENTS)))
    out_weights = np.zeros((n_days, n_assets))
    out_trades = np.zeros((n_days, n_assets))
    out_turnover = np.zeros(n_days)
    out_n_trades = np.zeros(n_days)
    out_participation = np.zeros(n_days)

    for d in range(n_days):
        price = exec_price[d]
        tradable = np.isfinite(price) & (price > 0)

        target_notional = targets[d] * nav
        trade = target_notional - pos

        if d % max(ex.rebalance_every, 1) == 0:
            # Skip churn below the no-trade band or below the size that can clear
            # the per-order commission minimum -- but never skip an exit, or
            # dropped-from-universe names accumulate as permanent stale risk.
            threshold = max(ex.no_trade_band * nav, ex.min_trade_dollars)
            exiting = (target_notional == 0.0) & (pos != 0.0)
            trade = np.where((np.abs(trade) >= threshold) | exiting, trade, 0.0)
        else:
            trade = np.zeros(n_assets)

        trade = np.where(tradable, trade, 0.0)

        safe_price = np.where(tradable, price, 1.0)
        trade_shares = trade / safe_price
        pos_after = pos + trade

        costs = cost_model.charge(
            trade_shares=trade_shares,
            trade_notional=trade,
            position_notional=pos_after,
            adv_shares=adv_shares[d],
            sigma=sigma[d],
            half_spread_bps=None if spread is None else spread[d],
            nav=nav,
        )

        gross_pnl = float(pos_after @ hold_ret[d])
        net_pnl = gross_pnl - costs.total

        out_nav[d] = nav
        out_gross[d] = gross_pnl / nav
        out_net[d] = net_pnl / nav
        out_costs[d] = [getattr(costs, c) for c in COMPONENTS]
        out_weights[d] = pos_after / nav
        out_trades[d] = trade
        out_turnover[d] = np.abs(trade).sum() / nav
        traded = trade != 0.0
        out_n_trades[d] = traded.sum()
        if traded.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                part = np.where(adv_shares[d] > 0, np.abs(trade_shares) / adv_shares[d], 0.0)
            out_participation[d] = float(np.nan_to_num(part[traded]).mean())

        nav += net_pnl
        if nav <= 0:
            log.error("NAV wiped out on %s; halting backtest", dates[d])
            out_nav[d + 1:] = 0.0
            break
        pos = pos_after * (1.0 + hold_ret[d])

    index = pd.DatetimeIndex(dates)
    weights_df = pd.DataFrame(out_weights, index=index, columns=tickers)
    exposures = pd.DataFrame(
        {
            "gross": weights_df.abs().sum(axis=1),
            "net": weights_df.sum(axis=1),
            "long": weights_df.clip(lower=0).sum(axis=1),
            "short": weights_df.clip(upper=0).sum(axis=1),
        }
    )

    return BacktestResult(
        nav=pd.Series(out_nav, index=index, name="nav"),
        gross_return=pd.Series(out_gross, index=index, name="gross_return"),
        net_return=pd.Series(out_net, index=index, name="net_return"),
        costs=pd.DataFrame(out_costs, index=index, columns=list(COMPONENTS)),
        weights=weights_df,
        target_weights=targets_df,
        trades=pd.DataFrame(out_trades, index=index, columns=tickers),
        turnover=pd.Series(out_turnover, index=index, name="turnover"),
        n_trades=pd.Series(out_n_trades, index=index, name="n_trades"),
        exposures=exposures,
        participation=pd.Series(out_participation, index=index, name="participation"),
        meta={"timing": ex.timing, "initial_nav": ex.initial_nav, "config": cfg.to_dict()},
    )
