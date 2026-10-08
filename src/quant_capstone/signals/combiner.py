"""Regime conditioning: trade the book bigger when the edge is stronger.

Motivated by a measurement, not a search. Blend IC across quartiles of trailing
cross-sectional return dispersion:

    Q1 (low)  -0.0028      Q3         +0.0181
    Q2        +0.0078      Q4 (high)  +0.0242

Monotone across all four buckets, and the lowest quartile is *negative* — the
strategy pays to trade in quiet cross-sections. Market realized volatility was
tested the same way and came out jagged (0.0117, 0.0109, -0.0002, 0.0249), so it
is treated as noise and not used.

**Why dispersion and not something else.** The reversal sleeves are paid for
providing liquidity, and the premium for immediacy is larger when stocks are
moving apart from one another. So the conditioning variable is the one the
mechanism predicts, which is why the per-sleeve breakdown is confirmatory rather
than decorative: both reversal sleeves strengthen monotonically with dispersion
while the earnings sleeve, which is not a liquidity-provision bet, shows no
pattern at all.

**Why this scales risk rather than re-weighting sleeves.** The risk pipeline
targets a fixed ex-ante volatility, so scaling the *signal* does nothing —
vol-targeting simply scales it straight back. The multiplier therefore applies to
the finished weights, and it only ever scales *down*. Scaling up would collide
with the per-name cap that already binds, and would be a leverage decision
dressed up as a forecasting one.

**On the DSR budget.** Three pre-registered specifications, chosen from the shape
of the table above before any backtest was run. Regime conditioning is the most
trial-hungry direction available and the Deflated Sharpe charges for every
configuration, so the discipline is to pick few and commit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_dispersion(
    returns: pd.DataFrame, members: pd.DataFrame | None = None,
    window: int = 21, min_periods: int = 10,
) -> pd.Series:
    """Trailing mean of daily cross-sectional return dispersion.

    Lagged one bar: dispersion measured through day t is known at the t close,
    and the weights it scales are traded at the t+1 open.
    """
    masked = returns.where(members) if members is not None else returns
    return masked.std(axis=1).rolling(window, min_periods=min_periods).mean().shift(1)


def regime_percentile(
    variable: pd.Series, lookback: int = 504, min_periods: int = 252
) -> pd.Series:
    """Where today's reading sits in its own trailing history, in [0, 1].

    An expanding/rolling percentile rather than a full-sample one: a fixed
    threshold fitted to the whole history would be hindsight, and the level of
    dispersion drifts enough over a decade that an absolute cutoff would misread
    whole years.
    """
    return variable.rolling(lookback, min_periods=min_periods).rank(pct=True)


def regime_multiplier(
    variable: pd.Series,
    mode: str = "linear",
    floor: float = 0.4,
    cap: float = 1.0,
    lookback: int = 504,
) -> pd.Series:
    """Risk multiplier in ``[floor, cap]`` from a regime variable.

    ``linear``  multiplier rises linearly with the trailing percentile.
    ``binary``  full size above the median, ``floor`` below it.
    ``off``     always 1.0, the control.

    Before enough history exists to rank against, the multiplier is 1.0 — the
    unconditional book — so the overlay can never act on an unformed estimate.
    """
    if mode == "off":
        return pd.Series(1.0, index=variable.index)

    percentile = regime_percentile(variable, lookback)
    if mode == "linear":
        multiplier = floor + (cap - floor) * percentile
    elif mode == "binary":
        multiplier = pd.Series(
            np.where(percentile >= 0.5, cap, floor), index=variable.index, dtype=float
        )
    else:
        raise ValueError(f"unknown regime mode {mode!r}")

    return multiplier.fillna(1.0).clip(floor, cap)


def regime_diagnostics(multiplier: pd.Series, returns: pd.Series) -> pd.Series:
    """Did the overlay actually move risk toward the profitable periods?"""
    aligned = pd.concat([multiplier.rename("m"), returns.rename("r")], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    high = aligned[aligned["m"] >= aligned["m"].median()]
    low = aligned[aligned["m"] < aligned["m"].median()]
    return pd.Series(
        {
            "mean_multiplier": float(aligned["m"].mean()),
            "days_at_floor": float((aligned["m"] <= aligned["m"].min() + 1e-9).mean()),
            "mean_return_when_scaled_up": float(high["r"].mean()),
            "mean_return_when_scaled_down": float(low["r"].mean()),
            "return_gap": float(high["r"].mean() - low["r"].mean()),
        }
    )
