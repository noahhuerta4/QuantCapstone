"""Cost sensitivity and the break-even point (spec S10, S6).

Raise costs until the edge dies. The ratio of the break-even cost level to the
baseline is the strategy's **robustness margin**, and it is one of the more
useful numbers the whole project produces: a strategy that breaks even at 1.2x
assumed costs is one bad slippage estimate away from being worthless, while one
that survives to 4x has genuine room.

This is cheap to compute because costs do not feed back into position sizing —
the weights are identical at every cost level, so only the backtest engine needs
re-running, not the signal or the portfolio construction. If you ever add
cost-aware optimisation (a no-trade band tuned to the cost level, say), that
stops being true and these curves have to be rebuilt from ``run_strategy``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from ..backtest.engine import run_backtest
from ..costs.model import COMPONENTS
from ..reporting.metrics import sharpe

if TYPE_CHECKING:  # importing StrategyRun eagerly would close an import cycle
    from ..strategy import StrategyRun

DEFAULT_MULTIPLIERS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)


def sweep_cost_multiplier(
    run: "StrategyRun", multipliers: Iterable[float] = DEFAULT_MULTIPLIERS
) -> pd.DataFrame:
    """Net Sharpe and return as a function of a global scale on every cost."""
    rows = []
    for m in multipliers:
        cfg = run.cfg.override(costs__cost_multiplier=float(m))
        result = run_backtest(run.dataset, run.portfolio.weights, cfg)
        rows.append(
            {
                "cost_multiplier": float(m),
                "gross_sharpe": sharpe(result.gross_return),
                "net_sharpe": sharpe(result.net_return),
                "net_ann_return": float(result.net_return.mean() * 252),
                "cost_bps_of_nav_annual": float(
                    result.costs.sum(axis=1).div(result.nav).mean() * 252 * 1e4
                ),
            }
        )
    return pd.DataFrame(rows).set_index("cost_multiplier")


def break_even_multiplier(sweep: pd.DataFrame, target_sharpe: float = 0.0) -> float:
    """Linearly interpolate the cost multiplier at which net Sharpe hits a target."""
    series = sweep["net_sharpe"].dropna()
    if series.empty or series.iloc[0] <= target_sharpe:
        return np.nan

    below = series[series <= target_sharpe]
    if below.empty:
        return np.inf  # survives the entire swept range

    hi = below.index[0]
    above = series[series.index < hi]
    if above.empty:
        return float(hi)
    lo = above.index[-1]

    y_lo, y_hi = series[lo], series[hi]
    if y_lo == y_hi:
        return float(hi)
    return float(lo + (target_sharpe - y_lo) * (hi - lo) / (y_hi - y_lo))


def sweep_component(
    run: "StrategyRun", field: str, values: Iterable[float]
) -> pd.DataFrame:
    """Sweep a single cost parameter, e.g. ``half_spread_bps`` or ``impact_coef``."""
    rows = []
    for value in values:
        cfg = run.cfg.override(**{f"costs__{field}": value})
        result = run_backtest(run.dataset, run.portfolio.weights, cfg)
        rows.append({field: value, "net_sharpe": sharpe(result.net_return)})
    return pd.DataFrame(rows).set_index(field)


def component_toggle_table(run: "StrategyRun") -> pd.DataFrame:
    """Net Sharpe with each cost component switched off in turn.

    The drop from 'all costs' to 'without slippage' is the honest measure of how
    much the slippage assumption is carrying — and on a daily reversal book it
    will dominate everything else put together.
    """
    rows = []
    scenarios = {"all costs on": {}, "no costs at all": {f"enable_{c}": False for c in COMPONENTS}}
    for component in COMPONENTS:
        scenarios[f"without {component}"] = {f"enable_{component}": False}

    for name, overrides in scenarios.items():
        cfg = run.cfg.override(**{f"costs__{k}": v for k, v in overrides.items()})
        result = run_backtest(run.dataset, run.portfolio.weights, cfg)
        rows.append(
            {
                "scenario": name,
                "net_sharpe": sharpe(result.net_return),
                "annual_cost_bps": float(
                    result.costs.sum(axis=1).div(result.nav).mean() * 252 * 1e4
                ),
            }
        )
    return pd.DataFrame(rows).set_index("scenario")


def capacity_sweep(run: "StrategyRun", navs: Iterable[float]) -> pd.DataFrame:
    """Net Sharpe as a function of fund size.

    Two things bite as NAV grows and both are modelled: the liquidity cap
    (1% of ADV) becomes a binding position constraint, and square-root impact
    rises with participation. This is the sweep that says how much money the
    strategy can actually run, which is usually a more interesting answer than
    the Sharpe itself.
    """
    from ..risk.pipeline import build_portfolio

    rows = []
    for nav in navs:
        cfg = run.cfg.override(execution__initial_nav=float(nav))
        portfolio = build_portfolio(run.dataset, run.signal, cfg)
        result = run_backtest(run.dataset, portfolio.weights, cfg)
        rows.append(
            {
                "nav": float(nav),
                "net_sharpe": sharpe(result.net_return),
                "gross_sharpe": sharpe(result.gross_return),
                "mean_gross_exposure": float(result.exposures["gross"].mean()),
                "mean_participation": float(
                    result.participation[result.participation > 0].mean()
                ),
                "annual_cost_bps": float(
                    result.costs.sum(axis=1).div(result.nav).mean() * 252 * 1e4
                ),
            }
        )
    return pd.DataFrame(rows).set_index("nav")
