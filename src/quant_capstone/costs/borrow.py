"""Stock borrow on the short leg (spec S6).

A persistent drag rather than a per-trade cost: it accrues on the short market
value every day the position is open, so it scales with *gross exposure and
holding period*, not with turnover. At a 10% vol target this book can run 2x
gross (1x short), and 50 bps/yr on 1x short notional is ~50 bps/yr of NAV —
material against a strategy whose gross Sharpe implies maybe 10% annual return.

We charge the general-collateral rate uniformly. Real life is worse: the names a
reversal signal most wants to short are recent big winners, which skew toward
harder-to-borrow. Treat this term as a floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import TRADING_DAYS, CostConfig


@dataclass(frozen=True)
class BorrowModel:
    annual_rate: float

    @classmethod
    def from_config(cls, cfg: CostConfig) -> "BorrowModel":
        return cls(cfg.borrow_rate)

    def charge(self, short_notional: float) -> float:
        """One day's borrow in dollars on the (positive) short market value.

        Dividing an annual rate by 252 trading days, not 360 calendar days:
        borrow really accrues over weekends, but our engine only steps on trading
        days, so the per-trading-day rate has to carry the full annual cost.
        """
        return float(self.annual_rate / TRADING_DAYS * max(short_notional, 0.0))

    def charge_per_name(self, notional: np.ndarray) -> np.ndarray:
        """Per-name borrow, for attribution. ``notional`` is signed."""
        return self.annual_rate / TRADING_DAYS * np.maximum(-notional, 0.0)
