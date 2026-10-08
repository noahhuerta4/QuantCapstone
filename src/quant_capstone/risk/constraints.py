"""Hard position constraints (spec S5.5).

Per-name caps, a liquidity cap, and a gross leverage cap. The tension worth
understanding: clipping a weight to its cap breaks neutrality (the clipped amount
has to go somewhere), and re-projecting to restore neutrality can push another
weight back over its cap. ``risk/pipeline.py`` alternates the two to a fixed
point, then finishes with a *uniform* down-scale, which is the only operation
that tightens every constraint at once without disturbing neutrality.

The liquidity cap is where strategy capacity shows up. At NAV of $10m against a
name with $200m ADV, a 1%-of-ADV cap is 20% of NAV -- so far above the 1.5%
per-name cap that it never binds. At $1bn NAV the same cap is 0.2% of NAV and
binds on every position. Same strategy, and the second one cannot be run. Use
``binding_constraint_report`` to see which cap is actually doing the work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import RiskConfig


def per_name_caps(
    adv_dollars: np.ndarray, reference_nav: float, cfg: RiskConfig
) -> np.ndarray:
    """Tightest per-name weight cap: the fixed cap or the liquidity cap."""
    with np.errstate(divide="ignore", invalid="ignore"):
        liquidity_cap = cfg.max_adv_participation * adv_dollars / max(reference_nav, 1.0)
    liquidity_cap = np.nan_to_num(liquidity_cap, nan=0.0, posinf=cfg.max_weight)
    return np.minimum(cfg.max_weight, liquidity_cap)


def clip_to_caps(weights: np.ndarray, caps: np.ndarray) -> np.ndarray:
    return np.clip(weights, -caps, caps)


def cap_gross(weights: np.ndarray, max_gross: float) -> np.ndarray:
    """Uniform down-scale if gross leverage exceeds the cap. Preserves neutrality."""
    gross = np.abs(weights).sum()
    if gross <= max_gross or gross <= 0:
        return weights
    return weights * (max_gross / gross)


def enforce(weights: np.ndarray, caps: np.ndarray, max_gross: float) -> np.ndarray:
    """Final safety pass: a single uniform scale that makes every cap hold.

    Uniform because neutrality has already been imposed by this point and only a
    uniform scale leaves it intact. The price is that one name pressed against
    its cap shrinks the whole book -- which is honest, and shows up as a lower
    transfer coefficient rather than as a silently violated constraint.
    """
    if not weights.size:
        return weights
    # A name with no usable ADV has a cap of zero; drop it outright rather than
    # letting an infinite breach ratio scale the entire book to nothing.
    weights = np.where(caps > 0, weights, 0.0)

    breach = np.max(np.abs(weights) / np.maximum(caps, 1e-12))
    scale = 1.0 / breach if breach > 1.0 else 1.0

    gross = np.abs(weights).sum()
    if gross > 0 and gross * scale > max_gross:
        scale = max_gross / gross
    return weights * scale


def binding_constraint_report(
    weights: pd.DataFrame, adv_dollars: pd.DataFrame, reference_nav: float, cfg: RiskConfig
) -> pd.DataFrame:
    """Per-day count of positions sitting at each cap. Diagnoses capacity."""
    liquidity_cap = (cfg.max_adv_participation * adv_dollars / max(reference_nav, 1.0))
    held = weights.abs() > 1e-10
    at_name_cap = (weights.abs() >= cfg.max_weight * 0.999) & held
    at_liquidity_cap = (weights.abs() >= liquidity_cap.reindex_like(weights) * 0.999) & held

    return pd.DataFrame(
        {
            "positions": held.sum(axis=1),
            "at_name_cap": at_name_cap.sum(axis=1),
            "at_liquidity_cap": at_liquidity_cap.sum(axis=1),
            "gross": weights.abs().sum(axis=1),
            "gross_cap": cfg.max_gross,
        }
    )
