"""signal -> weights, one explicit stage at a time (spec S5).

Order of operations, and why:

1. **Alpha to raw weights** (inverse-vol or mean-variance).
2. **Project** onto the neutrality null space. Done before scaling because the
   projection changes the portfolio's risk.
3. **Scale** to the ex-ante vol target. Uniform, so neutrality survives.
4. **Clip** to per-name and liquidity caps. This breaks neutrality and lowers
   risk, so...
5. **Repeat 2-4** to a fixed point. Each pass moves less than the last.
6. **Project once more, then uniformly down-scale** until every cap holds. This
   is the only finish that guarantees both neutrality and the caps exactly. It
   can leave the book below its vol target — which is real, and the right place
   for that shortfall to appear is the transfer coefficient, not a fudge.

We also keep the pre-constraint ``ideal_weights``. Their correlation with the
delivered weights *is* the transfer coefficient in Section 7, so this is the one
place it can be measured honestly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..data.loader import Dataset
from ..factors import market_return, rolling_beta, sector_dummies
from .constraints import cap_gross, clip_to_caps, enforce, per_name_caps
from .covariance import CovarianceEstimator
from .neutralize import neutrality_loadings
from .sizing import mean_variance, portfolio_vol, risk_weighted, scale_to_vol_target
from ..linalg import project_out

log = logging.getLogger(__name__)

MIN_NAMES = 20


@dataclass
class PortfolioOutput:
    weights: pd.DataFrame
    ideal_weights: pd.DataFrame
    """Pre-constraint, pre-neutralization target. Only used to measure TC."""
    diagnostics: pd.DataFrame
    beta: pd.DataFrame

    @property
    def transfer_correlation(self) -> pd.Series:
        """Daily cross-sectional correlation of delivered vs ideal weights."""
        return self.weights.corrwith(self.ideal_weights, axis=1)


def build_portfolio(dataset: Dataset, signal: pd.DataFrame, cfg: Config) -> PortfolioOutput:
    prices, members = dataset.prices, dataset.members
    rc = cfg.risk
    dates, tickers = prices.dates, prices.tickers
    n_days, n_assets = len(dates), len(tickers)

    returns = prices.returns()
    beta_df = rolling_beta(returns, market_return(returns, members), cfg.signal.beta_lookback)
    dummies = sector_dummies(prices.sectors, tickers)
    adv_dollars = prices.adv(cfg.data.adv_lookback)
    reference_nav = cfg.execution.initial_nav

    signal_np = signal.reindex(index=dates, columns=tickers).to_numpy(dtype=float)
    members_np = members.reindex(index=dates, columns=tickers).fillna(False).to_numpy()
    beta_np = beta_df.to_numpy(dtype=float)
    adv_np = np.nan_to_num(adv_dollars.to_numpy(dtype=float), nan=0.0)

    estimator = CovarianceEstimator(rc)

    weights = np.zeros((n_days, n_assets))
    ideal = np.zeros((n_days, n_assets))
    diag_rows = np.zeros((n_days, 5))

    for d in range(n_days):
        active = members_np[d] & np.isfinite(signal_np[d])
        if int(active.sum()) < MIN_NAMES:
            continue
        candidates = np.flatnonzero(active)

        # Sigma is estimated over the names tradable on the refresh day; anything
        # that has joined the universe since then has no estimate yet and waits.
        scope = candidates if rc.cov_scope == "universe" else np.arange(n_assets)
        estimator.refresh_if_stale(returns, d, scope, n_assets)
        cov, available = estimator.block(candidates)
        idx = candidates[available]
        n_active = len(idx)
        if n_active < MIN_NAMES:
            continue

        alpha = signal_np[d, idx]

        raw = (
            mean_variance(alpha, cov, rc.mvo_lambda)
            if rc.weight_mode == "mvo"
            else risk_weighted(alpha, cov)
        )
        ideal[d, idx] = scale_to_vol_target(raw, cov, rc.vol_target)

        loadings = neutrality_loadings(
            tickers[idx],
            pd.Series(beta_np[d, idx], index=tickers[idx]),
            dummies.iloc[idx],
            rc,
        )
        caps = per_name_caps(adv_np[d, idx], reference_nav, rc)

        w = raw
        for _ in range(max(rc.constraint_iters, 1)):
            w = project_out(w, loadings)
            w = scale_to_vol_target(w, cov, rc.vol_target)
            w = clip_to_caps(w, caps)
            w = cap_gross(w, rc.max_gross)
        w = project_out(w, loadings)
        w = enforce(w, caps, rc.max_gross)

        weights[d, idx] = w
        diag_rows[d] = [
            n_active,
            portfolio_vol(w, cov),
            np.abs(w).sum(),
            w.sum(),
            int((np.abs(w) >= caps * 0.999).sum()),
        ]

    weights_frame = pd.DataFrame(weights, index=pd.DatetimeIndex(dates), columns=tickers)
    if rc.regime_scaling != "off":
        from ..signals.combiner import cross_sectional_dispersion, regime_multiplier

        dispersion = cross_sectional_dispersion(returns, members)
        multiplier = regime_multiplier(
            dispersion, rc.regime_scaling, rc.regime_floor, 1.0, rc.regime_lookback
        )
        # Uniform per-day scale: leaves dollar/beta/sector neutrality intact and
        # can only tighten the caps, never breach them.
        weights_frame = weights_frame.mul(multiplier.reindex(weights_frame.index).fillna(1.0), axis=0)
        log.info("regime overlay (%s): mean multiplier %.3f",
                 rc.regime_scaling, float(multiplier.mean()))

    index = pd.DatetimeIndex(dates)
    diagnostics = pd.DataFrame(
        diag_rows,
        index=index,
        columns=["n_active", "ex_ante_vol", "gross", "net", "n_at_cap"],
    )
    log.info(
        "portfolio: mean ex-ante vol %.2f%%, mean gross %.2fx, mean names %.0f",
        diagnostics.loc[diagnostics.n_active > 0, "ex_ante_vol"].mean() * 100,
        diagnostics.loc[diagnostics.n_active > 0, "gross"].mean(),
        diagnostics.loc[diagnostics.n_active > 0, "n_active"].mean(),
    )

    return PortfolioOutput(
        weights=weights_frame,
        ideal_weights=pd.DataFrame(ideal, index=index, columns=tickers),
        diagnostics=diagnostics,
        beta=beta_df,
    )
