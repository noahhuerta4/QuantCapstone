"""Neutralization — what actually makes the book market-neutral (spec S5.3).

Dollar, beta and sector neutrality are all the same constraint in different
clothing: ``L' w = 0`` for a loading matrix ``L`` whose columns are, respectively,
a vector of ones, the betas, and the sector indicators. So we stack them and take
one orthogonal projection, which satisfies all of them simultaneously and
exactly.

Doing them sequentially would not work: demeaning to get dollar neutrality
destroys the beta neutrality you just imposed, and vice versa. The joint
projection is not a nicety, it is the correct operation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import RiskConfig
from ..linalg import build_loadings, project_out


def neutrality_loadings(
    tickers: pd.Index,
    beta: pd.Series | None,
    sector_loadings: pd.DataFrame | None,
    cfg: RiskConfig,
) -> np.ndarray | None:
    """Stack the enabled neutrality constraints into one loading matrix."""
    blocks: list[np.ndarray] = []
    n = len(tickers)

    if cfg.neutralize_dollar:
        blocks.append(np.ones((n, 1)))
    if cfg.neutralize_beta and beta is not None:
        blocks.append(np.nan_to_num(beta.reindex(tickers).to_numpy(dtype=float), nan=1.0).reshape(-1, 1))
    if cfg.neutralize_sector and sector_loadings is not None:
        dummies = sector_loadings.reindex(index=tickers).fillna(0.0).to_numpy(dtype=float)
        # Sector dummies sum to the intercept; drop one to keep L full rank.
        # (project_out uses lstsq and tolerates rank deficiency, but a full-rank
        # L makes the residual easier to reason about and the tests exact.)
        if cfg.neutralize_dollar and dummies.shape[1] > 1:
            dummies = dummies[:, 1:]
        blocks.append(dummies)

    return build_loadings(*blocks)


def neutralize(weights: np.ndarray, loadings: np.ndarray | None) -> np.ndarray:
    """Project the weights onto the null space of the constraints."""
    return project_out(weights, loadings)


def neutrality_report(
    weights: pd.DataFrame, beta: pd.DataFrame, sectors: pd.Series
) -> pd.DataFrame:
    """Ex-post check that the constraints actually hold in the delivered book."""
    from ..factors import sector_dummies

    dummies = sector_dummies(sectors, weights.columns)
    beta_aligned = beta.reindex_like(weights).fillna(1.0)

    out = pd.DataFrame(index=weights.index)
    out["dollar_net"] = weights.sum(axis=1)
    out["beta_net"] = (weights * beta_aligned).sum(axis=1)
    sector_net = weights.to_numpy() @ dummies.to_numpy()
    out["max_abs_sector_net"] = np.abs(sector_net).max(axis=1)
    return out
