"""Regulatory fees on sales (spec S6). Small, but a market-neutral book sells
roughly half its turnover every day, so they are not zero.

Both fees are charged on *sales only* — including short sales, which is why a
long/short book pays them on about half of its gross turnover rather than none
of it. Rates change; verify before quoting results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CostConfig


@dataclass(frozen=True)
class RegulatoryFees:
    sec_fee_rate: float
    """SEC Section 31, per dollar of sale proceeds."""
    finra_taf_per_share: float
    finra_taf_cap: float
    """FINRA TAF is capped per trade."""

    @classmethod
    def from_config(cls, cfg: CostConfig) -> "RegulatoryFees":
        return cls(cfg.sec_fee_rate, cfg.finra_taf_per_share, cfg.finra_taf_cap)

    def charge(self, signed_shares: np.ndarray, signed_notional: np.ndarray) -> np.ndarray:
        """Per-name fees in dollars. Negative share deltas are sales."""
        is_sale = signed_shares < 0
        sale_shares = np.where(is_sale, np.abs(signed_shares), 0.0)
        sale_notional = np.where(is_sale, np.abs(signed_notional), 0.0)

        sec = self.sec_fee_rate * sale_notional
        taf = np.minimum(self.finra_taf_per_share * sale_shares, self.finra_taf_cap)
        return sec + np.where(sale_shares > 0, taf, 0.0)
