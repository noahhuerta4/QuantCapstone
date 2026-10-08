"""Blend alpha sleeves into one score.

The blend is where the diversification benefit is realised or thrown away, and
the choices that matter are about robustness, not cleverness.

**Standardize before combining.** Each sleeve is z-scored cross-sectionally
first, so a raw 12-month momentum number (order 0.3) and a 2-day reversal number
(order 0.01) contribute equally instead of momentum silently owning the book.

**Equal weights by default.** Estimating sleeve weights from realized IC is the
obvious next step and usually a mistake: it is another parameter set fitted on
the same sample, and at PBO 0.79 on *two* parameters, fitting six more is asking
for it. Equal weighting of standardized sleeves is the robust choice, and the
burden of proof is on any departure from it. ``ic_weighted`` exists for the
comparison, computes its weights from a trailing window only, and is off by
default.

**Blend at the signal level, not the portfolio level.** Combining scores before
constructing the portfolio lets opposing sleeve views net out *before* they reach
the market — if momentum wants to buy a name that reversal wants to sell, we
trade the difference rather than paying spread on both legs. That netting is
worth more than the flexibility of running separate sub-portfolios.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .transforms import standardize, winsorize_mad


def prepare_sleeve(
    raw: pd.DataFrame, members: pd.DataFrame, winsorize: bool = True,
    n_mad: float = 3.0, method: str = "zscore",
) -> pd.DataFrame:
    """Mask to the universe, tame outliers, standardize. Per sleeve, per day."""
    masked = raw.where(members.reindex_like(raw).fillna(False))
    if winsorize:
        masked = winsorize_mad(masked, n_mad)
    return standardize(masked, method)


def equal_weight(sleeves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean of the standardized sleeves, ignoring sleeves that are NaN that day.

    Using the mean of *available* sleeves rather than a fixed sum matters during
    warm-up: momentum needs 273 bars of history and reversal needs 5, so early in
    the sample only some sleeves exist, and summing would silently shrink the
    combined score toward zero exactly when the fast sleeves are all we have.
    """
    if not sleeves:
        raise ValueError("no sleeves to combine")
    stacked = np.stack([s.to_numpy(dtype=float) for s in sleeves.values()])
    with warnings.catch_warnings():
        # Days before any sleeve has warmed up are legitimately all-NaN.
        warnings.simplefilter("ignore", RuntimeWarning)
        combined = np.nanmean(stacked, axis=0)
    template = next(iter(sleeves.values()))
    return pd.DataFrame(combined, index=template.index, columns=template.columns)


def fixed_weight(sleeves: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """Weighted mean of standardized sleeves, renormalized over available ones."""
    names = list(sleeves)
    w = np.array([float(weights.get(n, 0.0)) for n in names])
    if not np.isfinite(w).all() or np.abs(w).sum() == 0:
        raise ValueError("sleeve weights must be finite and not all zero")

    stacked = np.stack([sleeves[n].to_numpy(dtype=float) for n in names])
    present = np.isfinite(stacked)
    weighted = np.where(present, stacked, 0.0) * w[:, None, None]
    denominator = (present * np.abs(w)[:, None, None]).sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        combined = np.where(denominator > 0, weighted.sum(axis=0) / denominator, np.nan)
    template = sleeves[names[0]]
    return pd.DataFrame(combined, index=template.index, columns=template.columns)


def ic_weighted(
    sleeves: dict[str, pd.DataFrame], forward: pd.DataFrame,
    lookback: int = 504, min_periods: int = 252, refresh: int = 21,
    floor: float = 0.0,
) -> pd.DataFrame:
    """Weight each sleeve by its own trailing IC. Off by default -- see module doc.

    Strictly point-in-time: the weight applied on day t is estimated from IC
    realized strictly before t. Weights are floored at ``floor`` (zero by
    default) so a sleeve that has been unprofitable is dropped rather than
    shorted, which would be a much larger bet than the evidence supports.
    """
    from ..linalg import rank_corr

    names = list(sleeves)
    template = sleeves[names[0]]
    n_days = len(template)

    # Daily IC per sleeve, then a trailing mean, lagged so day t sees only t-1.
    fwd = forward.reindex_like(template).to_numpy(dtype=float)
    ic = np.full((n_days, len(names)), np.nan)
    for j, name in enumerate(names):
        values = sleeves[name].to_numpy(dtype=float)
        ic[:, j] = [rank_corr(values[i], fwd[i]) for i in range(n_days)]

    ic_frame = pd.DataFrame(ic, index=template.index, columns=names)
    trailing = ic_frame.rolling(lookback, min_periods=min_periods).mean().shift(1)
    # Hold weights between refreshes so they do not add turnover of their own.
    trailing = trailing.iloc[::refresh].reindex(template.index, method="ffill")
    weights = trailing.clip(lower=floor).fillna(0.0)

    stacked = np.stack([sleeves[n].to_numpy(dtype=float) for n in names])
    w = weights.to_numpy(dtype=float).T[:, :, None]
    present = np.isfinite(stacked)
    numerator = (np.where(present, stacked, 0.0) * w).sum(axis=0)
    denominator = (present * np.abs(w)).sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        combined = np.where(denominator > 0, numerator / denominator, np.nan)
    # Before any sleeve has enough history to be scored, fall back to equal weights.
    fallback = equal_weight(sleeves).to_numpy(dtype=float)
    combined = np.where(np.isfinite(combined), combined, fallback)
    return pd.DataFrame(combined, index=template.index, columns=template.columns)


def sleeve_correlations(sleeves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean daily cross-sectional correlation between sleeves.

    The number that decides whether combining is worth anything. Sleeves
    correlated at 0.8 are one sleeve wearing two hats and add no breadth;
    near-zero correlations are what turn N sleeves into sqrt(N) of Sharpe.
    """
    names = list(sleeves)
    out = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            value = float(sleeves[a].corrwith(sleeves[b], axis=1).mean())
            out.loc[a, b] = out.loc[b, a] = value
    return out


def diversification_ratio(correlations: pd.DataFrame) -> float:
    """Sharpe multiple from combining N equally-weighted sleeves.

    ``sqrt(N / (1 + (N-1) * rho_bar))`` -- the factor by which the blend's IR
    exceeds a single sleeve's, given the average pairwise correlation. At rho = 0
    it is sqrt(N); at rho = 1 it is 1 and the blend has bought nothing.
    """
    n = len(correlations)
    if n < 2:
        return 1.0
    off_diagonal = correlations.to_numpy()[~np.eye(n, dtype=bool)]
    rho = float(np.nanmean(off_diagonal))
    denominator = 1.0 + (n - 1) * rho
    return float(np.sqrt(n / denominator)) if denominator > 0 else np.nan
