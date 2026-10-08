"""Transfer coefficient — how much of the ideal alpha survives (spec S7).

Clarke, de Silva and Thorley's TC is the cross-sectional correlation between the
weights you *wanted* (unconstrained, from alpha and risk) and the weights you
*took* after constraints. It is bounded by 1 and is where caps, neutrality and
liquidity limits get charged.

Costs are not in that correlation — they hit the return, not the weights — so we
measure them separately as ``net Sharpe / gross Sharpe`` and report the product.
Splitting them is the point: TC_constraints says the portfolio construction is
throwing away alpha, TC_costs says the trading is. They have completely different
remedies, and conflating them into one number tells you nothing about which to
fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def transfer_coefficient(ideal: pd.DataFrame, actual: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional correlation of delivered weights with ideal ones."""
    aligned_ideal = ideal.reindex(index=actual.index, columns=actual.columns)
    both_live = (actual.abs().sum(axis=1) > 0) & (aligned_ideal.abs().sum(axis=1) > 0)
    tc = actual.corrwith(aligned_ideal, axis=1)
    return tc.where(both_live).rename("transfer_coefficient")


def cost_transfer(gross_sharpe: float, net_sharpe: float) -> float:
    """Fraction of the gross Sharpe that survives trading costs.

    Negative net Sharpe gives a negative ratio, which is the honest reading: the
    costs did not merely erode the edge, they inverted it.
    """
    if not np.isfinite(gross_sharpe) or gross_sharpe == 0:
        return np.nan
    return float(net_sharpe / gross_sharpe)


def transfer_summary(
    ideal: pd.DataFrame, actual: pd.DataFrame, gross_sharpe: float, net_sharpe: float
) -> pd.Series:
    tc_series = transfer_coefficient(ideal, actual).dropna()
    tc_constraints = float(tc_series.mean()) if len(tc_series) else np.nan
    tc_costs = cost_transfer(gross_sharpe, net_sharpe)
    return pd.Series(
        {
            "tc_constraints": tc_constraints,
            "tc_constraints_std": float(tc_series.std(ddof=1)) if len(tc_series) > 1 else np.nan,
            "tc_costs": tc_costs,
            "tc_total": tc_constraints * tc_costs,
        }
    )
