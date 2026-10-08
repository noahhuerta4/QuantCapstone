"""Optional factor orthogonalization of the raw signal (spec S4).

Why bother, when Section 5 neutralizes the *weights* anyway? Because neutralizing
at the weight stage removes the factor bet but leaves the alpha ranking distorted
by it — a day when all the recent losers happen to be high-beta names produces a
signal whose cross-sectional ordering is partly a beta ordering. Orthogonalizing
first means the ranking itself is a cleaner statement about idiosyncratic
reversion, and the neutralization downstream then has less work to do (which
shows up as a better transfer coefficient).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..linalg import build_loadings, project_out


def orthogonalize(
    signal: pd.DataFrame,
    beta: pd.DataFrame | None = None,
    sector_loadings: pd.DataFrame | None = None,
    intercept: bool = True,
) -> pd.DataFrame:
    """Cross-sectionally regress the signal on the given loadings, keep residuals."""
    values = signal.to_numpy(dtype=float)
    n_assets = values.shape[1]

    beta_values = beta.reindex_like(signal).to_numpy(dtype=float) if beta is not None else None
    sector_matrix = (
        sector_loadings.reindex(index=signal.columns).to_numpy(dtype=float)
        if sector_loadings is not None
        else None
    )
    ones = np.ones((n_assets, 1)) if intercept else None

    out = np.full_like(values, np.nan)
    for i in range(values.shape[0]):
        row = values[i]
        present = np.isfinite(row)
        if present.sum() < 5:
            continue
        blocks = [ones, sector_matrix]
        if beta_values is not None:
            blocks.append(beta_values[i].reshape(-1, 1))
        loadings = build_loadings(*[b for b in blocks if b is not None])
        if loadings is None:
            out[i] = row
            continue
        # Restrict to present names so absent ones do not enter the regression.
        resid = np.full(n_assets, np.nan)
        resid[present] = project_out(row[present], loadings[present])
        out[i] = resid

    return pd.DataFrame(out, index=signal.index, columns=signal.columns)
