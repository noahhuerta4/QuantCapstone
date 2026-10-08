"""Slippage and market impact — the dominant cost term (spec S6).

Two models behind one flag:

``half_spread``
    Charge a fixed number of bps per side. Crossing the spread to get filled
    costs you roughly half the quoted spread; for S&P 500 names that is ~1-3 bps.
    Size-independent, which is its weakness: it says a $50k trade and a $5m trade
    in the same name cost the same in bps, and they do not.

``impact`` (default)
    Half-spread plus a square-root impact term, ``c * sigma * sqrt(shares/ADV)``.
    The square-root law is the standard empirical finding (Almgren et al. 2005;
    Torre's BARRA work before it): impact grows with the square root of
    participation, scaled by the name's own volatility.

Calibrating ``c``: with sigma ~ 2%/day and the S5 liquidity cap holding
participation at 1% of ADV, ``c = 0.1`` gives 0.1 * 0.02 * 0.1 = 2 bps of impact,
comparable to the half-spread. Published estimates of the coefficient in this
parameterisation cluster in the 0.05-0.5 range depending on venue and horizon, so
0.1 is a mid-conservative choice — and the cost-sensitivity sweep in S10 exists
precisely because we do not get to be confident about it. Report the break-even
``c``, not just the base case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CostConfig

BPS = 1e-4


@dataclass(frozen=True)
class SlippageModel:
    half_spread_bps: float
    impact_coef: float
    use_impact: bool

    @classmethod
    def from_config(cls, cfg: CostConfig) -> "SlippageModel":
        if cfg.slippage_model not in ("half_spread", "impact"):
            raise ValueError(f"unknown slippage model {cfg.slippage_model!r}")
        return cls(cfg.half_spread_bps, cfg.impact_coef, cfg.slippage_model == "impact")

    def rate(
        self,
        shares: np.ndarray,
        adv_shares: np.ndarray,
        sigma: np.ndarray,
        half_spread_bps: np.ndarray | None = None,
    ) -> np.ndarray:
        """Slippage as a fraction of notional, per name.

        ``half_spread_bps`` overrides the configured flat rate with a per-name
        estimate (see ``costs/spread.py``). That matters as soon as the universe
        spans more than mega-caps: charging every name the same 2 bps would make
        illiquid names look artificially cheap to trade, which is precisely the
        assumption a liquidity study must not make.
        """
        if half_spread_bps is None:
            rate = np.full(np.shape(shares), self.half_spread_bps * BPS, dtype=float)
        else:
            # Fall back to the configured rate wherever the estimate is missing.
            estimate = np.asarray(half_spread_bps, dtype=float)
            rate = np.where(np.isfinite(estimate), estimate, self.half_spread_bps) * BPS
        if not self.use_impact:
            return rate

        with np.errstate(divide="ignore", invalid="ignore"):
            participation = np.where(adv_shares > 0, np.abs(shares) / adv_shares, 0.0)
        participation = np.nan_to_num(participation, nan=0.0, posinf=0.0)
        sigma = np.nan_to_num(sigma, nan=0.0)
        return rate + self.impact_coef * sigma * np.sqrt(participation)

    def charge(
        self,
        shares: np.ndarray,
        notional: np.ndarray,
        adv_shares: np.ndarray,
        sigma: np.ndarray,
        half_spread_bps: np.ndarray | None = None,
    ) -> np.ndarray:
        """Per-name slippage in dollars."""
        return self.rate(shares, adv_shares, sigma, half_spread_bps) * np.abs(notional)
