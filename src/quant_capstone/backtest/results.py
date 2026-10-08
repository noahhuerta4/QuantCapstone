"""Container for everything one backtest produced.

Deliberately fat: the whole point of the spec is that we can answer 'where did
the Sharpe go' afterwards, and that needs the trade log and the cost split, not
just an equity curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class BacktestResult:
    nav: pd.Series
    gross_return: pd.Series
    net_return: pd.Series
    costs: pd.DataFrame
    """Dollar costs per day, one column per component."""
    weights: pd.DataFrame
    """Actual post-trade weights (fraction of NAV), not targets."""
    target_weights: pd.DataFrame
    trades: pd.DataFrame
    """Signed dollar notional traded."""
    turnover: pd.Series
    """One-way traded notional as a fraction of NAV."""
    n_trades: pd.Series
    exposures: pd.DataFrame
    """gross / net / long / short, as fractions of NAV."""
    participation: pd.Series
    """Mean fraction of ADV consumed by a traded name, for impact sanity checks."""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def cost_return(self) -> pd.Series:
        """Total cost as a fraction of prior NAV — gross_return minus this is net."""
        return self.gross_return - self.net_return

    def slice(self, start=None, end=None) -> "BacktestResult":
        """Restrict to a date window (used by walk-forward to carve out OOS)."""
        sl = slice(start, end)
        return BacktestResult(
            nav=self.nav.loc[sl],
            gross_return=self.gross_return.loc[sl],
            net_return=self.net_return.loc[sl],
            costs=self.costs.loc[sl],
            weights=self.weights.loc[sl],
            target_weights=self.target_weights.loc[sl],
            trades=self.trades.loc[sl],
            turnover=self.turnover.loc[sl],
            n_trades=self.n_trades.loc[sl],
            exposures=self.exposures.loc[sl],
            participation=self.participation.loc[sl],
            meta={**self.meta, "sliced": (str(start), str(end))},
        )

    def daily_frame(self) -> pd.DataFrame:
        """Flat per-day table, handy for eyeballing and for CSV export."""
        return pd.concat(
            [
                self.nav.rename("nav"),
                self.gross_return.rename("gross_return"),
                self.net_return.rename("net_return"),
                self.cost_return.rename("cost_return"),
                self.turnover.rename("turnover"),
                self.n_trades.rename("n_trades"),
                self.exposures,
                self.costs.add_prefix("cost_"),
            ],
            axis=1,
        )
