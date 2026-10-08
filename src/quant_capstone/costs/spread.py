"""Per-name bid-ask spread for the cost model.

A flat half-spread in bps is fine for a universe of mega-caps that all trade
around a basis point. It is indefensible once the universe spans the liquidity
spectrum — a small-cap can trade at 30-50 bps, twenty times the large-cap figure,
and assuming otherwise decides any liquidity comparison by assumption.

**Which model to use, and a negative result worth knowing.** This module offers
three. The default, ``liquidity_scaled_half_spread``, is a calibrated function of
dollar volume. The other two — Corwin-Schultz and Abdi-Ranaldo — estimate the
spread from the bars themselves, which is more principled in general and *does
not work here*. Measured against mega-caps whose true half-spreads are common
knowledge (AAPL ~0.3 bps, MSFT ~0.5 bps), they return:

    ticker   true    Corwin-Schultz   Abdi-Ranaldo
    AAPL    ~0.3        0.00              4.91
    MSFT    ~0.5        4.73             19.90
    XOM     ~0.5        0.00             12.02

Off by an order of magnitude, in both directions, with no usable ordering. The
reason is signal-to-noise: both estimators infer the spread from the geometry of
the daily range, and for a liquid US name the spread is ~0.3 bps against ~150 bps
of daily volatility. There is nothing to recover at that ratio — the estimate is
measuring volatility. These estimators were designed for markets where the spread
is a material fraction of daily movement (emerging markets, micro-caps, pre-
decimalisation US data), and this universe is the opposite regime.

They are kept, documented, and available for a wider universe, but they are not
the default and their output should not be trusted for liquid names.

The estimator uses the insight that a two-day high-low range contains two days of
volatility but only *one* spread, while two separate one-day ranges contain two
days of volatility and two spreads. Differencing isolates the spread:

    beta  = (ln(H_t/L_t))^2 + (ln(H_t+1/L_t+1))^2      two single-day ranges
    gamma = (ln(max(H_t,H_t+1) / min(L_t,L_t+1)))^2    one two-day range
    alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2))
            - sqrt(gamma / (3 - 2*sqrt(2)))
    S     = 2 * (exp(alpha) - 1) / (1 + exp(alpha))

``S`` is the proportional spread, so the half-spread is ``S/2``.

Two practical notes. Negative estimates occur routinely on quiet days and are
floored at zero, per the paper. And the estimate for day t reads day t+1, so it
is not known until t+1 closes -- ``rolling_half_spread_bps`` handles that lag
explicitly rather than leaving a one-day look-ahead in the cost model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DENOM = 3.0 - 2.0 * np.sqrt(2.0)
BPS = 1e-4


#: Log-log calibration of half-spread against dollar volume, anchored to
#: published US equity spread levels: ~0.35 bps at $1bn ADV, ~1.9 at $100m,
#: ~4.6 at $30m, ~10 at $10m, ~25 at $3m.
SPREAD_COEFFICIENT = 56.0
SPREAD_EXPONENT = -0.735
SPREAD_FLOOR_BPS = 0.2
SPREAD_CAP_BPS = 100.0


def liquidity_scaled_half_spread(adv_dollars: pd.DataFrame) -> pd.DataFrame:
    """Half-spread in bps as a calibrated function of dollar volume.

    ``half_spread_bps = 56 * ADV_in_millions ** -0.735``, floored and capped.

    This is an **assumption calibrated to published spread levels**, not an
    estimate from our data — and that is the honest description of it. We do not
    have quote data, and the bar-based estimators demonstrably cannot resolve
    spreads at this liquidity (see the module docstring). A transparent,
    monotone, literature-anchored function beats a noisy estimator that is wrong
    by 10x in both directions.

    Two consequences worth stating. It is smooth in ADV, so it cannot manufacture
    a spurious liquidity effect through estimation noise. And because it is an
    assumption, the cost-sensitivity sweep is doing real work rather than
    decorating: the break-even multiplier is the number that carries the
    uncertainty here, not the base case.
    """
    adv_millions = (adv_dollars / 1e6).replace(0.0, np.nan)
    raw = SPREAD_COEFFICIENT * adv_millions.pow(SPREAD_EXPONENT)
    return raw.clip(lower=SPREAD_FLOOR_BPS, upper=SPREAD_CAP_BPS)


def corwin_schultz_spread(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """Raw proportional spread estimate. Row t uses days t and t+1."""
    h = np.log(high.where(high > 0))
    l = np.log(low.where(low > 0))

    single = (h - l) ** 2
    beta = single + single.shift(-1)

    two_day_high = np.maximum(h, h.shift(-1))
    two_day_low = np.minimum(l, l.shift(-1))
    gamma = (two_day_high - two_day_low) ** 2

    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / DENOM - np.sqrt(gamma / DENOM)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return spread.clip(lower=0.0)


def abdi_ranaldo_spread(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
    window: int = 21, min_periods: int = 5,
) -> pd.DataFrame:
    """Proportional spread from close and the high-low mid (Abdi & Ranaldo 2017).

    Uses the gap between the close and the *mid-range* on consecutive days:

        eta_t = (ln H_t + ln L_t) / 2
        s^2   = 4 * (c_t - eta_t) * (c_t - eta_t+1)

    Preferred to Corwin-Schultz here because it is materially less biased when
    prices gap overnight. CS attributes the whole two-day high-low range to
    spread, so an overnight jump — which is information, not a transaction cost —
    inflates its estimate. That bias lands hardest on exactly the mega-caps whose
    true spreads are near a basis point, which is where our universe lives.

    Averaged inside the square root, not outside: the estimator is unbiased for
    ``s^2``, not for ``s``, and squaring per-day then taking one root at the end
    is what the paper specifies.
    """
    h, l = np.log(high.where(high > 0)), np.log(low.where(low > 0))
    c = np.log(close.where(close > 0))
    eta = (h + l) / 2.0

    s_squared = 4.0 * (c - eta) * (c - eta.shift(-1))
    mean_sq = s_squared.rolling(window, min_periods=min_periods).mean()
    return np.sqrt(mean_sq.clip(lower=0.0))


def rolling_half_spread_bps(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame | None = None,
    window: int = 21, min_periods: int = 5, estimator: str = "abdi_ranaldo",
) -> pd.DataFrame:
    """Smoothed, point-in-time half-spread in basis points.

    ``abdi_ranaldo`` is the default because Corwin-Schultz is badly biased upward
    when prices gap overnight; it needs ``close`` and falls back to CS without it.

    The ``shift(2)`` is the look-ahead control: both estimators read day t+1 at
    row t, so a value stamped at t is knowable only after the t+1 close, and a
    signal formed on day t may use it no earlier than t+2.
    """
    if estimator == "abdi_ranaldo" and close is not None:
        raw = abdi_ranaldo_spread(high, low, close, window, min_periods)
        smoothed = raw.shift(2)          # already averaged inside the estimator
    elif estimator in ("abdi_ranaldo", "corwin_schultz"):
        # Median, not mean: the daily CS estimate is noisy and right-skewed, so a
        # single wide-range day would otherwise own the window.
        raw = corwin_schultz_spread(high, low)
        smoothed = raw.rolling(window, min_periods=min_periods).median().shift(2)
    else:
        raise ValueError(f"unknown spread estimator {estimator!r}")
    return (smoothed / 2.0) / BPS


def spread_summary(half_spread_bps: pd.DataFrame) -> pd.Series:
    """Cross-sectional distribution of the estimate, for sanity checking.

    Expect single digits for mega-caps and tens of bps for small-caps. If the
    whole distribution comes out flat, the estimator is not working and the cost
    model is back to assuming its answer.
    """
    values = half_spread_bps.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return pd.Series(dtype=float)
    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return pd.Series(
        {
            "mean_bps": float(values.mean()),
            "p05_bps": float(quantiles[0]),
            "p25_bps": float(quantiles[1]),
            "median_bps": float(quantiles[2]),
            "p75_bps": float(quantiles[3]),
            "p95_bps": float(quantiles[4]),
            "pct_zero": float((values <= 0).mean()),
        }
    )
