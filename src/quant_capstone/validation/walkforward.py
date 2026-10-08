"""Walk-forward out-of-sample evaluation (spec S10).

Fit the parameter choice on a train window, evaluate on the next untouched
window, roll forward, and stitch the untouched windows into one OOS series. Only
that stitched series counts.

Two comparisons make the output worth reading:

* **Selected vs fixed-default.** If picking the best parameters per fold does no
  better OOS than just using the defaults everywhere, the selection is fitting
  noise — a genuinely useful and slightly deflating thing to learn.
* **Train Sharpe vs test Sharpe, per fold.** The average gap is the overfitting
  premium in the units anyone cares about.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..config import Config
from ..data.loader import Dataset
from ..reporting.metrics import sharpe, summarize
from .grid import RunCache, label

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __str__(self) -> str:
        return f"{self.test_start.date()}..{self.test_end.date()}"


def make_folds(
    dates: pd.DatetimeIndex, train_days: int, test_days: int, expanding: bool = True,
    embargo: int = 0,
) -> list[Fold]:
    """Consecutive non-overlapping test windows, each preceded by its own train.

    ``embargo`` drops that many days from the *end* of each training window
    (Lopez de Prado's purging and embargoing). It matters here because positions
    are held for ten days: a trade opened near the end of training is still open
    into the test window, so its return straddles the boundary and appears in
    both. Without an embargo the train and test sets share outcomes, and the
    measured generalisation gap is flattered by the overlap.
    """
    folds: list[Fold] = []
    train_end = train_days
    while train_end + test_days <= len(dates):
        train_lo = 0 if expanding else train_end - train_days
        # Purge the holding period off the end of training, not off the start of
        # testing -- the test window must stay a contiguous, untouched block.
        purged_end = train_end - max(embargo, 0)
        if purged_end - max(train_lo, 0) < test_days:
            break
        folds.append(
            Fold(
                train_start=dates[max(train_lo, 0)],
                train_end=dates[purged_end - 1],
                test_start=dates[train_end],
                test_end=dates[min(train_end + test_days, len(dates)) - 1],
            )
        )
        train_end += test_days
    return folds


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame
    oos_returns: pd.Series
    """Stitched out-of-sample net returns from the selected configurations."""
    oos_returns_default: pd.Series
    cache: RunCache
    metric: str

    def summary(self) -> pd.DataFrame:
        return pd.concat(
            [
                summarize(self.oos_returns, "oos_selected"),
                summarize(self.oos_returns_default, "oos_default"),
            ],
            axis=1,
        )

    def stability(self) -> pd.Series:
        """Is the OOS Sharpe sustained across windows, or one lucky period?"""
        test_sharpes = self.folds["test_sharpe"].dropna()
        return pd.Series(
            {
                "n_folds": float(len(test_sharpes)),
                "mean_test_sharpe": float(test_sharpes.mean()),
                "median_test_sharpe": float(test_sharpes.median()),
                "worst_test_sharpe": float(test_sharpes.min()),
                "best_test_sharpe": float(test_sharpes.max()),
                "folds_positive": float((test_sharpes > 0).mean()),
                "folds_above_2": float((test_sharpes >= 2.0).mean()),
                "mean_train_minus_test": float(
                    (self.folds["train_sharpe"] - self.folds["test_sharpe"]).mean()
                ),
            }
        )


def walk_forward(
    base: Config,
    dataset: Dataset,
    grid: dict[str, Iterable[Any]],
    cache: RunCache | None = None,
    net: bool = True,
) -> WalkForwardResult:
    cache = cache or RunCache(dataset)
    # Always evaluate the un-overridden base config, so the "fixed defaults"
    # comparison column really is the defaults. Without this, a non-empty grid
    # leaves no entry named ``label({})`` and the fallback below silently used
    # whichever cell happened to come first -- reporting an arbitrary grid
    # corner under a heading that claims to be the default.
    names = cache.evaluate(base, {}) + cache.evaluate(base, grid)
    names = list(dict.fromkeys(names))
    returns = cache.returns(names, net=net)

    vc = base.validation
    # Embargo the holding period so open positions cannot straddle the boundary.
    embargo = max(vc.embargo_days, base.execution.rebalance_every)
    folds = make_folds(returns.index, vc.train_days, vc.test_days, vc.expanding, embargo)
    if not folds:
        raise ValueError(
            f"history too short for walk-forward: need > {vc.train_days + vc.test_days} days, "
            f"have {len(returns.index)}"
        )

    default_name = label({})
    if default_name not in returns.columns:  # should not happen; guard anyway
        log.warning("base configuration missing from the run cache; comparing against %s", names[0])
        default_name = names[0]

    rows, oos_chunks, default_chunks = [], [], []
    for fold in folds:
        train = returns.loc[fold.train_start : fold.train_end]
        test = returns.loc[fold.test_start : fold.test_end]

        train_sharpes = train.apply(sharpe)
        if not np.isfinite(train_sharpes).any():
            continue
        chosen = str(train_sharpes.idxmax())

        rows.append(
            {
                "fold": str(fold),
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "selected": chosen,
                "train_sharpe": float(train_sharpes[chosen]),
                "test_sharpe": sharpe(test[chosen]),
                "test_sharpe_default": sharpe(test[default_name]),
                "n_train_days": len(train),
                "n_test_days": len(test),
            }
        )
        oos_chunks.append(test[chosen].rename("oos"))
        default_chunks.append(test[default_name].rename("oos"))

    return WalkForwardResult(
        folds=pd.DataFrame(rows),
        oos_returns=pd.concat(oos_chunks) if oos_chunks else pd.Series(dtype=float),
        oos_returns_default=pd.concat(default_chunks) if default_chunks else pd.Series(dtype=float),
        cache=cache,
        metric="net_sharpe" if net else "gross_sharpe",
    )


def in_sample_vs_out_of_sample(result: WalkForwardResult, name: str | None = None) -> pd.DataFrame:
    """IS and OOS side by side for one configuration (spec S0: report both)."""
    name = name or label({})
    run = result.cache.runs[name]
    oos_index = result.oos_returns.index
    full = run.result

    is_mask = ~full.net_return.index.isin(oos_index)
    return pd.concat(
        [
            summarize(full.gross_return[is_mask], "IS_gross"),
            summarize(full.net_return[is_mask], "IS_net"),
            summarize(full.gross_return.reindex(oos_index), "OOS_gross"),
            summarize(full.net_return.reindex(oos_index), "OOS_net"),
        ],
        axis=1,
    )
