"""Breadth — and why the naive count is a lie (spec S7).

The seduction: rebalancing N=200 names every day looks like BR = 252*200 =
50,400 independent bets a year. The Fundamental Law then says IR = IC*sqrt(BR),
so IR = 2 needs only IC = 2/sqrt(50400) = 0.009. A correlation of nine
thousandths. Free money, apparently.

Three things are wrong with it, and this module measures all three.

**1. Names are not independent.** Equity returns share a dominant market factor
plus sector factors. The participation ratio of the correlation matrix's
eigenvalues, ``(sum L)^2 / sum L^2``, counts how many genuinely distinct
directions there are; for 200 large-caps it is typically under 20. A
market-neutral book has already projected out the dominant factor, so the honest
count for *this* strategy is the participation ratio of the **residual**
correlation matrix after removing the top factors — higher than the raw figure,
but still far below N.

**2. Holding the same tilt is not a fresh bet.** A k-day reversal signal has
lag-1 autocorrelation around 1 - 1/k. Standard AR(1) effective-sample-size says T
autocorrelated observations carry the information of ``T * (1-rho)/(1+rho)``
independent ones. At rho = 0.8 that turns 252 rebalances into about 28.

**3. TC then knocks it down again** — see ``transfer.py``.

Multiply the two corrections and the 50,400 becomes a few hundred. The IC needed
for IR = 2 rises by more than an order of magnitude, into territory where you
have to ask whether the signal really has it.

**An honest caveat on the second correction.** The AR(1) adjustment is the
effective sample size for estimating the mean of an autocorrelated series, and
applying it to breadth implicitly assumes the daily IC is the IC of a *new*
forecast. It is not: a k-day reversal signal holds a similar position for several
days and earns its edge on each of them, so pairing a one-day-ahead IC with
252/k periods undercounts by roughly sqrt(k). The literature is genuinely not
settled here (compare Grinold-Kahn's original treatment with Qian-Hua's
adjustment). Rather than pick a side and hide it, ``breadth_implied_by_performance``
inverts the law against the realized result, so the report can show the
theoretical estimate and the demonstrated one next to each other and let the gap
speak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS


def participation_ratio(correlation: np.ndarray) -> float:
    """Effective number of independent directions in a correlation matrix.

    ``(sum lambda)^2 / sum lambda^2``. Equals N for the identity (all independent)
    and 1 for a matrix of all ones (one bet wearing N hats).
    """
    eig = np.linalg.eigvalsh(correlation)
    eig = np.clip(eig, 0.0, None)
    denom = float((eig**2).sum())
    return float(eig.sum() ** 2 / denom) if denom > 0 else np.nan


def residual_correlation(returns: pd.DataFrame, n_factors: int = 1) -> np.ndarray:
    """Correlation matrix after stripping the top ``n_factors`` principal components.

    This is the matrix a neutralized book actually bets in: the market factor has
    been projected out of the weights, so its contribution to bet dependence is
    not the strategy's to worry about.
    """
    # Drop the leading all-NaN return row first: dropping *columns* with any NaN
    # while that row is present would discard the entire panel.
    clean = returns.dropna(axis=0, how="all")
    clean = clean.loc[:, clean.notna().mean() >= 0.8]
    if clean.shape[1] < 3 or clean.shape[0] < 10:
        return np.eye(max(clean.shape[1], 1))

    centered = clean - clean.mean()
    standardized = centered / centered.std(ddof=1).replace(0.0, np.nan)
    # Remaining gaps become zero deviation from the mean, contributing nothing.
    matrix = standardized.fillna(0.0).to_numpy(dtype=float)

    if n_factors > 0:
        _, _, vt = np.linalg.svd(matrix, full_matrices=False)
        loadings = vt[:n_factors].T
        matrix = matrix - matrix @ loadings @ loadings.T

    corr = np.corrcoef(matrix, rowvar=False)
    return np.nan_to_num(corr, nan=0.0)


def autocorrelation_adjustment(rho: float) -> float:
    """Fraction of rebalances that count as independent, for an AR(1) signal."""
    rho = float(np.clip(rho, -0.999, 0.999))
    return (1.0 - rho) / (1.0 + rho)


def effective_breadth(
    returns: pd.DataFrame,
    signal: pd.DataFrame,
    members: pd.DataFrame,
    n_factors_removed: int = 1,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """Naive vs effective breadth, with the two corrections separated out."""
    in_universe = members.any()
    universe_returns = returns.loc[:, in_universe[in_universe].index]
    mean_n = float(members.sum(axis=1).replace(0, np.nan).mean())

    # The participation ratio is computed over every name that was *ever* in the
    # universe, which is more names than are held on any given day. Convert it to
    # an independence fraction and re-express it at the typical daily universe
    # size, so breadth reflects the book actually run.
    raw_corr = np.nan_to_num(universe_returns.corr().to_numpy(dtype=float), nan=0.0)
    n_cols = max(raw_corr.shape[0], 1)
    n_eff_raw = participation_ratio(raw_corr) / n_cols * mean_n
    residual = residual_correlation(universe_returns, n_factors_removed)
    n_eff_residual = participation_ratio(residual) / max(residual.shape[0], 1) * mean_n

    rho = float(signal.corrwith(signal.shift(1), axis=1).mean())
    time_factor = autocorrelation_adjustment(rho)

    naive = mean_n * periods_per_year
    effective = n_eff_residual * periods_per_year * time_factor

    return pd.Series(
        {
            "mean_names": mean_n,
            "n_eff_raw_correlation": n_eff_raw,
            "n_eff_after_factor_removal": n_eff_residual,
            "signal_autocorr_lag1": rho,
            "independent_period_fraction": time_factor,
            "effective_periods_per_year": periods_per_year * time_factor,
            "breadth_naive": naive,
            "breadth_effective": effective,
            "breadth_shrinkage": effective / naive if naive > 0 else np.nan,
        }
    )


def required_ic(target_ir: float, breadth: float, transfer_coefficient: float = 1.0) -> float:
    """Invert the Fundamental Law: what IC does this IR demand?"""
    if breadth <= 0 or transfer_coefficient <= 0:
        return np.nan
    return float(target_ir / (transfer_coefficient * np.sqrt(breadth)))


def breadth_implied_by_performance(realized_ir: float, ic: float, tc: float = 1.0) -> float:
    """Invert the law against what actually happened: ``BR = (IR / (TC*IC))^2``.

    Free of any assumption about how to count bets, because it does not count
    them -- it reads the number off the result. Compare it to ``breadth_naive``
    (it will be far smaller) and to ``breadth_effective`` (where the direction of
    the gap tells you whether the correction above was too harsh or too kind).
    """
    if not np.isfinite(ic) or ic == 0 or not np.isfinite(tc) or tc == 0:
        return np.nan
    return float((realized_ir / (tc * ic)) ** 2)


def seductive_lie_table(breadth: pd.Series, target_ir: float = 2.0, tc: float = 1.0) -> pd.DataFrame:
    """Side-by-side of the naive and honest IC requirements for ``target_ir``."""
    rows = {
        "naive (BR = 252 x N, TC = 1)": {
            "breadth": breadth["breadth_naive"],
            "transfer_coefficient": 1.0,
            "required_ic": required_ic(target_ir, breadth["breadth_naive"], 1.0),
        },
        "correlation-adjusted": {
            "breadth": breadth["n_eff_after_factor_removal"] * TRADING_DAYS,
            "transfer_coefficient": 1.0,
            "required_ic": required_ic(
                target_ir, breadth["n_eff_after_factor_removal"] * TRADING_DAYS, 1.0
            ),
        },
        "+ autocorrelation-adjusted": {
            "breadth": breadth["breadth_effective"],
            "transfer_coefficient": 1.0,
            "required_ic": required_ic(target_ir, breadth["breadth_effective"], 1.0),
        },
        "+ transfer coefficient": {
            "breadth": breadth["breadth_effective"],
            "transfer_coefficient": tc,
            "required_ic": required_ic(target_ir, breadth["breadth_effective"], tc),
        },
    }
    return pd.DataFrame(rows).T.rename_axis(f"assumption (target IR = {target_ir})")
