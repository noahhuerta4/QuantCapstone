"""The assembled cost model: every trade pays commission + fees + slippage, and
every short pays borrow (spec S6).

Each component is individually switchable, and ``cost_multiplier`` scales the lot
— that pair is what makes the Section 10 cost-sensitivity sweep a one-liner.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS, CostConfig
from .borrow import BorrowModel
from .commissions import CommissionModel
from .fees import RegulatoryFees
from .slippage import SlippageModel

COMPONENTS = ("commission", "reg_fees", "slippage", "borrow", "financing")


@dataclass(frozen=True)
class CostBreakdown:
    commission: float = 0.0
    reg_fees: float = 0.0
    slippage: float = 0.0
    borrow: float = 0.0
    financing: float = 0.0

    @property
    def total(self) -> float:
        return sum(getattr(self, f.name) for f in fields(self))

    def __add__(self, other: "CostBreakdown") -> "CostBreakdown":
        return CostBreakdown(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})

    def as_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_series(self) -> pd.Series:
        return pd.Series({**self.as_dict(), "total": self.total})


class CostModel:
    """Charges one rebalance. Stateless; the engine owns the accumulation."""

    def __init__(self, cfg: CostConfig):
        self.cfg = cfg
        self.commission = CommissionModel.from_config(cfg)
        self.fees = RegulatoryFees.from_config(cfg)
        self.slippage = SlippageModel.from_config(cfg)
        self.borrow = BorrowModel.from_config(cfg)

    def charge(
        self,
        *,
        trade_shares: np.ndarray,
        trade_notional: np.ndarray,
        position_notional: np.ndarray,
        adv_shares: np.ndarray,
        sigma: np.ndarray,
        half_spread_bps: np.ndarray | None = None,
        nav: float | None = None,
    ) -> CostBreakdown:
        """Dollar costs for one day.

        ``trade_*`` are signed (negative = sale, which is what triggers reg fees).
        ``position_notional`` is the signed position held *after* trading, which
        is what borrow accrues on. ``half_spread_bps``, when given, replaces the
        flat configured spread with a per-name estimate.
        """
        cfg = self.cfg
        zero = 0.0

        commission = (
            float(self.commission.charge(trade_shares, trade_notional).sum())
            if cfg.enable_commission else zero
        )
        reg_fees = (
            float(self.fees.charge(trade_shares, trade_notional).sum())
            if cfg.enable_reg_fees else zero
        )
        slippage = (
            float(
                self.slippage.charge(
                    trade_shares, trade_notional, adv_shares, sigma, half_spread_bps
                ).sum()
            )
            if cfg.enable_slippage else zero
        )
        borrow = (
            float(self.borrow.charge_per_name(position_notional).sum())
            if cfg.enable_borrow else zero
        )

        # Margin interest on the debt-funded part of the long leg. Zero unless
        # long exposure exceeds NAV, so an unlevered book pays nothing.
        financing = zero
        if cfg.enable_financing and nav is not None:
            long_notional = float(np.maximum(position_notional, 0.0).sum())
            financed = max(long_notional - nav, 0.0)
            financing = cfg.financing_rate / TRADING_DAYS * financed

        m = cfg.cost_multiplier
        return CostBreakdown(
            commission * m, reg_fees * m, slippage * m, borrow * m, financing * m
        )


def attribution_bps(costs: pd.DataFrame, nav: pd.Series) -> pd.Series:
    """Annualized cost drag in bps of NAV, split by component (spec S11).

    This is the table that answers 'where did the Sharpe go': if slippage is 250
    bps/yr and commissions are 40, no amount of commission optimisation saves the
    strategy.
    """
    from ..config import TRADING_DAYS

    daily_frac = costs.div(nav, axis=0)
    annual = daily_frac.mean() * TRADING_DAYS * 1e4
    present = [c for c in COMPONENTS if c in annual.index]
    annual["total"] = annual[present].sum()
    return annual.rename("annual_bps_of_nav")
