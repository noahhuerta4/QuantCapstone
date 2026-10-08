"""IBKR US equity commissions (spec S6). VERIFY THE SCHEDULE AT BUILD TIME.

The per-order minimum is the part that shapes the strategy. At $0.35 minimum and
$0.0035/share, any order below 100 shares pays more than the per-share rate
implies — a $2,000 trade in a $200 stock (10 shares) pays 1.75 bps instead of
0.18 bps, ten times the headline rate. That is the direct argument for no-trade
bands and a minimum trade size (S8), and it is why fragmenting the book across
more names is not free.

We model one order per name per rebalance. Real execution would slice a large
order into children, but IBKR charges the minimum per *order*, not per child
fill, so one-order-per-name is the right granularity here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CostConfig


@dataclass(frozen=True)
class CommissionModel:
    per_share: float
    min_per_order: float
    max_pct_trade_value: float

    @classmethod
    def from_config(cls, cfg: CostConfig) -> "CommissionModel":
        if cfg.commission_model == "tiered":
            return cls(cfg.per_share_tiered, cfg.min_per_order_tiered, cfg.max_pct_trade_value)
        if cfg.commission_model == "fixed":
            return cls(cfg.per_share_fixed, cfg.min_per_order_fixed, cfg.max_pct_trade_value)
        raise ValueError(f"unknown commission model {cfg.commission_model!r}")

    def charge(self, shares: np.ndarray, notional: np.ndarray) -> np.ndarray:
        """Per-name commission in dollars. ``shares``/``notional`` are absolute."""
        shares = np.abs(shares)
        notional = np.abs(notional)
        traded = shares > 0

        cost = np.maximum(self.per_share * shares, self.min_per_order)
        # IBKR's 1%-of-trade-value cap overrides the per-order minimum, which is
        # what stops a 1-share order in a $3 stock from costing $0.35.
        cost = np.minimum(cost, self.max_pct_trade_value * notional)
        return np.where(traded, cost, 0.0)

    def effective_bps(self, shares: np.ndarray, notional: np.ndarray) -> np.ndarray:
        """Commission in bps of notional — the number worth staring at."""
        notional = np.abs(notional)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(notional > 0, self.charge(shares, notional) / notional * 1e4, 0.0)
