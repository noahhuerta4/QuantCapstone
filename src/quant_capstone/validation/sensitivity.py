"""Parameter sensitivity surfaces (spec S10).

What to look for: a **plateau**, not a peak. If Sharpe is 2.1 at k=5 and 0.4 at
k=4 and k=6, the 2.1 is an artefact of the grid, not a property of the strategy —
nothing about market microstructure changes discontinuously between a four- and
a six-day lookback. A broad region of similar performance means the effect is
real even if the best cell is lower.

``plateau_score`` puts a number on that so it does not depend on how the heatmap
is coloured.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS, Config
from ..data.loader import Dataset
from ..reporting.metrics import sharpe
from .grid import RunCache, label


def _row(run, params: dict[str, Any]) -> dict[str, Any]:
    result = run.result
    return {
        **params,
        "gross_sharpe": sharpe(result.gross_return),
        "net_sharpe": sharpe(result.net_return),
        "net_ann_return": float(result.net_return.mean() * TRADING_DAYS),
        "ann_turnover": float(result.turnover.mean() * TRADING_DAYS),
        "trades_per_day": float(result.n_trades.mean()),
        "mean_gross": float(result.exposures["gross"].mean()),
        "annual_cost_bps": float(result.costs.sum(axis=1).div(result.nav).mean() * TRADING_DAYS * 1e4),
    }


def sweep(
    base: Config,
    dataset: Dataset,
    grid: dict[str, Iterable[Any]],
    cache: RunCache | None = None,
) -> tuple[pd.DataFrame, RunCache]:
    """Run a full grid and tabulate the headline numbers for every cell."""
    cache = cache or RunCache(dataset)
    names = cache.evaluate(base, grid)
    rows = [_row(cache.runs[n], cache.params[n]) for n in names]
    table = pd.DataFrame(rows)
    return table, cache


def surface(table: pd.DataFrame, key_a: str, key_b: str, metric: str = "net_sharpe") -> pd.DataFrame:
    """Pivot a two-parameter sweep into a heatmap-shaped frame."""
    return table.pivot(index=key_a, columns=key_b, values=metric).sort_index()


def plateau_score(values: np.ndarray | pd.DataFrame, tolerance: float = 0.25) -> pd.Series:
    """How much of the swept region performs near the best cell.

    ``fraction_within_tolerance`` is the headline: a cliff scores near zero (only
    the peak itself is good), a plateau scores high. ``peak_over_median`` is the
    complementary read — a large value means the reported best is an outlier
    against its own grid.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return pd.Series(dtype=float)

    peak = float(finite.max())
    median = float(np.median(finite))
    threshold = peak * (1.0 - tolerance) if peak > 0 else peak * (1.0 + tolerance)

    return pd.Series(
        {
            "peak": peak,
            "median": median,
            "min": float(finite.min()),
            "fraction_within_tolerance": float((finite >= threshold).mean()),
            "fraction_positive": float((finite > 0).mean()),
            "peak_over_median": float(peak / median) if median != 0 else np.nan,
            "dispersion": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "n_cells": float(finite.size),
        }
    )


DEFAULT_GRID: dict[str, list[Any]] = {
    "signal__k": [1, 2, 3, 5, 10],
    "signal__pead_weight": [0.0, 0.10, 0.20],
    "execution__rebalance_every": [5, 10, 15],
    "execution__no_trade_band": [0.003, 0.006, 0.010],
}
"""Every parameter that was chosen by looking at results, re-selected per fold.

The rule this grid enforces: a value picked with full-sample hindsight and then
pinned in the config is inherited by every walk-forward fold, which quietly
turns the "out-of-sample" number back into an in-sample one. So each of these
was tuned at some point in the project, and each is therefore re-chosen from
training data alone in every fold:

* ``k`` -- the reversal lookback, adopted as 2 from a full-sample IC sweep.
  Its values are the spec's pre-registered set (S4), not a range centred on the
  number that sweep picked, so the grid does not smuggle the hindsight back in.
* ``pead_weight`` -- the earnings-surprise share of the blend.
* ``rebalance_every`` -- the holding period, which interacts with ``k``: a fast
  k=1 signal held for fifteen days is stale noise, a slow k=10 signal re-traded
  every five pays for turnover it does not need.
* ``no_trade_band`` -- re-tuned in Phase 2; 0.010 was the full-sample peak.

The cost is breadth of search: every cell is another trial the Deflated Sharpe
Ratio discounts, and the selection itself can fit noise. Whether it did is read
off the ``OOS_net_default`` column, which runs the fixed defaults through the
same windows."""
