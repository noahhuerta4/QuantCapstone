"""Independent checks on the engine (spec S9).

Three of them, in increasing order of how much they cost to run:

1. **Analytic replication.** With no-trade bands and minimum trade sizes turned
   off, the engine fully rebalances to target every day, so its gross return must
   equal the one-line vectorized expression ``(w.shift(1) * forward_ret).sum(1)``.
   That expression shares no code with the engine loop, so agreement to
   floating-point tolerance rules out an entire class of indexing and accounting
   bugs. This is the check that catches the most, and it needs no dependencies.

2. **Timing-sensitivity probe.** Shift the weights a day each way and confirm the
   result moves *violently*. A backtest that scores the same whether it trades on
   yesterday's, today's or tomorrow's signal is not using the timing it claims
   to, and its headline number means nothing.

   Do not expect the peeking variant to look *better*. For a reversal signal it
   is dramatically worse, and that is the correct behaviour: advancing the
   weights by a day makes the book trade at the day-t open on information from
   the day-t close, so it buys names that are about to fall during day t and then
   earns the open-to-open return that contains that very fall. It inherits the
   move it is betting against. The diagnostic is the *magnitude* of the shift,
   not its sign — which is why ``sensitivity`` below is an absolute value.

3. **vectorbt.** An independently written engine. Optional, because vectorbt
   pins a heavy dependency tree; install with ``uv sync --extra crosscheck``.

Discrepancies must be understood before any result is trusted.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config
from ..data.loader import Dataset
from ..reporting.metrics import sharpe
from .engine import _execution_frames, run_backtest

log = logging.getLogger(__name__)

TOLERANCE = 1e-9


def analytic_gross_returns(dataset: Dataset, weights: pd.DataFrame, cfg: Config) -> pd.Series:
    """``sum(w_held * realized_return)``, written independently of the engine."""
    prices = dataset.prices
    _, hold = _execution_frames(prices, cfg.execution.timing)
    held = weights.shift(1) if cfg.execution.timing == "next_open" else weights
    held = held.reindex(index=prices.dates, columns=prices.tickers).fillna(0.0)
    return (held * hold.fillna(0.0)).sum(axis=1).rename("analytic_gross")


def replication_check(dataset: Dataset, weights: pd.DataFrame, cfg: Config) -> pd.Series:
    """Compare the engine's gross return to the analytic expression."""
    # Full rebalancing so the engine's held weights equal the targets exactly;
    # with a no-trade band in place the two are legitimately different.
    exact = cfg.override(
        execution__no_trade_band=0.0,
        execution__min_trade_dollars=0.0,
        execution__rebalance_every=1,
    )
    engine = run_backtest(dataset, weights, exact).gross_return
    analytic = analytic_gross_returns(dataset, weights, exact)

    diff = (engine - analytic).abs()
    return pd.Series(
        {
            "max_abs_difference": float(diff.max()),
            "mean_abs_difference": float(diff.mean()),
            "correlation": float(engine.corr(analytic)),
            "engine_sharpe": sharpe(engine),
            "analytic_sharpe": sharpe(analytic),
            "passes": float(diff.max() < TOLERANCE),
        }
    )


def lookahead_probe(dataset: Dataset, weights: pd.DataFrame, cfg: Config) -> pd.Series:
    """Shift the weights a day each way; both must move the result substantially."""
    base_sharpe = sharpe(run_backtest(dataset, weights, cfg).net_return)
    advanced = sharpe(run_backtest(dataset, weights.shift(-1), cfg).net_return)
    lagged = sharpe(run_backtest(dataset, weights.shift(1), cfg).net_return)

    sensitivity = min(abs(advanced - base_sharpe), abs(lagged - base_sharpe))
    return pd.Series(
        {
            "sharpe_baseline": base_sharpe,
            "sharpe_advanced_one_day": advanced,
            "sharpe_lagged_one_day": lagged,
            "sensitivity": sensitivity,
            # A tiny response means the engine is not honouring its own timing
            # convention -- the weights may as well be random with respect to it.
            "passes": float(sensitivity > 0.1 * max(abs(base_sharpe), 0.5)),
        }
    )


def vectorbt_check(dataset: Dataset, weights: pd.DataFrame, cfg: Config) -> pd.Series:
    """Re-run the gross book through vectorbt, if it is installed."""
    try:
        import vectorbt as vbt
    except ImportError:
        log.info("vectorbt not installed; skipping (uv sync --extra crosscheck)")
        return pd.Series({"available": 0.0})

    prices = dataset.prices
    # Feed vectorbt the series we actually transact in, so its bar-to-bar return
    # matches our holding return: adjusted opens under next_open execution.
    price = (prices.open if cfg.execution.timing == "next_open" else prices.close).ffill()
    held = weights.shift(1) if cfg.execution.timing == "next_open" else weights
    held = held.reindex_like(price).fillna(0.0)

    valid = price.notna().all() & (held.abs().sum() > 0)
    price, held = price.loc[:, valid], held.loc[:, valid]

    try:
        portfolio = vbt.Portfolio.from_orders(
            close=price,
            size=held,
            size_type="targetpercent",
            group_by=True,
            cash_sharing=True,
            init_cash=cfg.execution.initial_nav,
            fees=0.0,
            slippage=0.0,
            freq="1D",
        )
        vbt_returns = portfolio.returns()
    except Exception as exc:  # API drift between vectorbt versions
        log.warning("vectorbt cross-check failed to run: %s", exc)
        return pd.Series({"available": 0.0, "error": 1.0})

    engine = run_backtest(dataset, weights, cfg.override(costs__cost_multiplier=0.0)).gross_return
    joined = pd.concat([engine.rename("engine"), vbt_returns.rename("vectorbt")], axis=1).dropna()
    return pd.Series(
        {
            "available": 1.0,
            "engine_sharpe": sharpe(joined["engine"]),
            "vectorbt_sharpe": sharpe(joined["vectorbt"]),
            "correlation": float(joined["engine"].corr(joined["vectorbt"])),
            "max_abs_difference": float((joined["engine"] - joined["vectorbt"]).abs().max()),
            "n_days_compared": float(len(joined)),
        }
    )


def crosscheck(run) -> pd.DataFrame:
    """All available checks for a completed :class:`StrategyRun`."""
    dataset, weights, cfg = run.dataset, run.portfolio.weights, run.cfg
    blocks = {
        "replication": replication_check(dataset, weights, cfg),
        "lookahead_probe": lookahead_probe(dataset, weights, cfg),
        "vectorbt": vectorbt_check(dataset, weights, cfg),
    }
    frame = pd.concat(blocks, axis=0).to_frame("value")
    frame.index.names = ["check", "measure"]
    return frame
