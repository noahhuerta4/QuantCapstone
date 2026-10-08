"""Assemble the configured signal.

Pipeline: build each sleeve -> mask to the universe, winsorize, standardize
(per sleeve) -> blend -> optional beta/sector orthogonalization -> standardize
again.

Standardizing *per sleeve before* blending is what makes the blend meaningful:
raw momentum is order 0.3 and raw 2-day reversal is order 0.01, so combining the
raw numbers would hand the entire book to momentum. Standardizing again *after*
blending restores unit cross-sectional dispersion, which the risk pipeline's
vol-targeting assumes.

Output is a tidy ``date x ticker`` matrix, NaN outside the point-in-time
universe. NaN means "not a candidate", never "score of zero".
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import Config
from ..data.loader import Dataset
from ..factors import market_return, rolling_beta, sector_dummies
from .alphas import ALPHAS, WARMUP
from .events import EVENT_ALPHAS, EVENT_REQUIRES
from .fundamental import FUNDAMENTAL_ALPHAS, FUNDAMENTAL_REQUIRES, market_cap
from .combine import equal_weight, fixed_weight, ic_weighted, prepare_sleeve
from .orthogonalize import orthogonalize
from .transforms import mask_to_universe, standardize

log = logging.getLogger(__name__)


def requested_alphas(cfg: Config) -> tuple[str, ...]:
    """Sleeves to build: the configured list, plus PEAD if it carries weight."""
    alphas = tuple(cfg.signal.alphas)
    if cfg.signal.pead_weight > 0 and "earnings_surprise" not in alphas:
        alphas += ("earnings_surprise",)
    return alphas


def build_sleeves(dataset: Dataset, cfg: Config) -> dict[str, pd.DataFrame]:
    """Every configured sleeve, standardized and masked to the universe."""
    prices, members = dataset.prices, dataset.members
    sc = cfg.signal
    alphas = requested_alphas(cfg)
    returns = prices.returns()
    market = market_return(returns, members)
    beta = rolling_beta(returns, market, sc.beta_lookback)

    # Arguments each sleeve needs, built once and shared.
    kwargs = {
        "reversal": dict(close=prices.close, k=sc.k),
        "momentum": dict(
            close=prices.close, lookback=sc.momentum_lookback, skip=sc.momentum_skip
        ),
        "residual_momentum": dict(
            returns=returns, beta=beta, market=market,
            lookback=sc.momentum_lookback, skip=sc.momentum_skip,
        ),
        "low_volatility": dict(returns=returns, lookback=sc.vol_lookback),
        "idiosyncratic_volatility": dict(
            returns=returns, beta=beta, market=market, lookback=sc.vol_lookback
        ),
        "volume_shock_reversal": dict(
            close=prices.close, volume=prices.volume, k=sc.k, baseline=sc.volume_baseline
        ),
        "overnight_reversal": dict(open_=prices.open, close=prices.close, k=sc.overnight_k),
    }

    cap = None
    fundamental_names = [name for name in alphas if name in FUNDAMENTAL_ALPHAS]
    if fundamental_names:
        if not dataset.fundamentals:
            raise ValueError(
                "a fundamental sleeve was requested but the dataset carries no "
                "fundamentals; load it via load_dataset() so they are fetched"
            )
        # Only the ratio sleeves need a market cap; PEAD does not, so do not
        # demand a shares panel that this configuration will never use.
        if any("shares" in FUNDAMENTAL_REQUIRES[n] for n in fundamental_names):
            cap = market_cap(prices.close_raw, dataset.fundamentals["shares"])

    sleeves: dict[str, pd.DataFrame] = {}
    for name in alphas:
        if name in EVENT_ALPHAS:
            for requirement in EVENT_REQUIRES[name]:
                if requirement == "form4":
                    if dataset.form4 is None or dataset.form4.empty:
                        raise ValueError(f"sleeve {name!r} needs Form 4 data; load via load_dataset()")
                elif requirement not in dataset.fundamentals:
                    raise ValueError(f"sleeve {name!r} needs missing concept {requirement!r}")
            raw = EVENT_ALPHAS[name](dataset, cfg)
        elif name in FUNDAMENTAL_ALPHAS:
            missing = [
                c for c in FUNDAMENTAL_REQUIRES[name] if c not in dataset.fundamentals
            ]
            if missing:
                raise ValueError(f"sleeve {name!r} needs missing concepts {missing}")
            raw = FUNDAMENTAL_ALPHAS[name](dataset.fundamentals, cap)
        elif name in ALPHAS:
            raw = ALPHAS[name](**kwargs[name])
        else:
            known = sorted(set(ALPHAS) | set(FUNDAMENTAL_ALPHAS) | set(EVENT_ALPHAS))
            raise ValueError(f"unknown alpha sleeve {name!r}; known: {known}")
        sleeves[name] = prepare_sleeve(
            raw, members, sc.winsorize, sc.winsor_mad, sc.standardize
        )
    return sleeves


def build_signal(dataset: Dataset, cfg: Config) -> pd.DataFrame:
    prices, members = dataset.prices, dataset.members
    sc = cfg.signal
    sleeves = build_sleeves(dataset, cfg)

    if sc.pead_weight > 0 and "earnings_surprise" in sleeves and len(sleeves) > 1:
        others = [n for n in sleeves if n != "earnings_surprise"]
        share = (1.0 - sc.pead_weight) / len(others)
        weights = {n: share for n in others}
        weights["earnings_surprise"] = sc.pead_weight
        signal = fixed_weight(sleeves, weights)
    elif sc.combine == "equal":
        signal = equal_weight(sleeves)
    elif sc.combine == "fixed":
        signal = fixed_weight(sleeves, dict(sc.sleeve_weights))
    elif sc.combine == "ic_weighted":
        from ..backtest.engine import forward_returns

        signal = ic_weighted(sleeves, forward_returns(prices, cfg.execution.timing))
    else:
        raise ValueError(f"unknown combine mode {sc.combine!r}")

    if sc.orthogonalize_beta:
        returns = prices.returns()
        beta = rolling_beta(returns, market_return(returns, members), sc.beta_lookback)
        signal = orthogonalize(
            signal, beta=beta, sector_loadings=sector_dummies(prices.sectors, prices.tickers)
        )

    signal = standardize(signal, sc.standardize)
    signal = mask_to_universe(signal, members)
    log.info(
        "signal: %d sleeves (%s), combine=%s, warm-up %d bars",
        len(sleeves), ", ".join(sleeves), sc.combine,
        max((WARMUP.get(n, 0) for n in sleeves), default=0),
    )
    return signal


def signal_diagnostics(signal: pd.DataFrame) -> pd.Series:
    """Sanity numbers to eyeball before trusting anything downstream."""
    row_sum = signal.sum(axis=1)
    return pd.Series(
        {
            "n_days_with_signal": int((signal.notna().sum(axis=1) > 0).sum()),
            "mean_names_per_day": float(signal.notna().sum(axis=1).mean()),
            "max_abs_cross_sectional_sum": float(row_sum.abs().max()),
            "mean_abs_score": float(signal.abs().mean().mean()),
            "mean_cross_sectional_std": float(signal.std(axis=1).mean()),
        }
    )
