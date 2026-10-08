"""Parameter grids, and the run cache that makes sweeps affordable.

The key observation that makes everything in ``validation/`` fast enough to be
usable: **the pipeline is causal**, so a configuration run once over the full
history produces exactly the same day-t weights as the same configuration run
over any window ending at t. Universe selection uses trailing ADV, the signal
uses trailing returns, covariance and beta use trailing windows, and the weights
do not depend on NAV (the liquidity cap references a fixed NAV by design). So a
walk-forward evaluation does not need to re-run the strategy per fold; it runs
each configuration once and slices the resulting return series.

This is an optimisation, not a shortcut — and it is only legitimate because of
that causality property. If you ever add a stage that peeks at the full sample
(a full-history z-score, a covariance fitted on everything, a universe chosen
with hindsight), this stops being valid and the walk-forward silently becomes
in-sample. ``tests/test_no_lookahead.py`` guards the property.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from ..config import Config
from ..data.loader import Dataset
from ..strategy import StrategyRun, run_strategy

log = logging.getLogger(__name__)


def grid_configs(base: Config, grid: dict[str, Iterable[Any]]) -> list[tuple[dict[str, Any], Config]]:
    """Expand a ``{"signal__k": [1,3,5]}`` grid into (params, config) pairs."""
    if not grid:
        return [({}, base)]
    keys = list(grid)
    combos = itertools.product(*(list(grid[k]) for k in keys))
    return [(dict(zip(keys, values)), base.override(**dict(zip(keys, values)))) for values in combos]


def label(params: dict[str, Any]) -> str:
    """Compact, stable name for a parameter combination."""
    if not params:
        return "base"
    return ", ".join(f"{k.split('__')[-1]}={v}" for k, v in sorted(params.items()))


@dataclass
class RunCache:
    """Runs each configuration at most once and keeps the results."""

    dataset: Dataset
    runs: dict[str, StrategyRun] = field(default_factory=dict)
    params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def evaluate(self, base: Config, grid: dict[str, Iterable[Any]]) -> list[str]:
        names = []
        for params, cfg in grid_configs(base, grid):
            name = label(params)
            names.append(name)
            if name not in self.runs:
                log.info("running configuration %s", name)
                self.runs[name] = run_strategy(cfg, self.dataset)
                self.params[name] = params
        return names

    @property
    def n_trials(self) -> int:
        """Configurations tried — feeds the Deflated Sharpe Ratio honestly."""
        return len(self.runs)

    def returns(self, names: list[str] | None = None, net: bool = True) -> pd.DataFrame:
        names = names or list(self.runs)
        series = {
            n: (self.runs[n].result.net_return if net else self.runs[n].result.gross_return)
            for n in names
        }
        return pd.DataFrame(series)
