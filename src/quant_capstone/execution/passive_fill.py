"""Parametric model of passive execution, and the feasibility map it produces.

The question this answers before any microstructure work is attempted: *how good
would our fills have to be* for intraday trading to beat the daily book? If no
plausible fill quality clears it, the expensive Stage-2 simulation is not worth
building.

**The thesis being tested.** Hourly reversal carries roughly twenty times the
effective breadth of the daily signal, which is the term the daily book is short
of. It was set aside because the turnover needed to harvest it costs more than
the edge is worth. But that arithmetic assumes we *cross* the spread on every
trade. A resting limit order that gets hit **earns** the half-spread instead of
paying it, so the dominant turnover-linked cost changes sign rather than merely
shrinking. This is cost inversion, not cost reduction, and it is the only
mechanism that plausibly makes 20x turnover affordable.

**The two parameters.**

``p`` — fill ratio. The fraction of intended volume that fills passively. The
remainder does not simply vanish: to hold the target weights we cross it at the
end of the bar and pay the spread. So ``p`` interpolates between "pay the spread
on everything" (p = 0) and "earn it on everything" (p = 1), and the break-even
on the spread term alone sits at p = 0.5.

``a`` — adverse selection, in units of per-bar volatility. The catch that makes
passive execution harder than it looks: a resting buy order fills precisely when
sellers are motivated, which is disproportionately when the price is about to
keep falling. So filled orders realize a forward return worse than intended by
``a · σ_bar``. This term is what usually kills naive passive-execution
arithmetic, and it scales with turnover exactly as the spread saving does — so
the two fight each other directly.

**What the model deliberately does not do.** It has no queue, no order book, no
per-name heterogeneity, and it assumes the fill ratio is independent of the
signal. Every one of those simplifications flatters passive execution. Treat the
output as an upper bound and a screening device, not a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExecutionEconomics:
    """Everything the feasibility map needs, all measured rather than assumed.

    Defaults are the values measured on the daily book and the hourly study, so
    the map is anchored to this project rather than to generic parameters.
    """

    implied_gross_ir: float = 0.97
    """Hourly signal's implied IR after the measured transfer coefficient."""
    book_vol_bps: float = 370.0
    """Annualized volatility of the book, in basis points."""
    half_spread_bps: float = 0.44
    """Calibrated per-name half-spread for this universe."""
    bar_vol_bps: float = 65.0
    """Per-bar volatility, measured on the hourly panel (cross-sectionally
    demeaned, as the book actually trades it). Sets the scale of adverse selection."""
    bars_per_year: float = 1764.0
    """Roughly 7 hourly bars x 252 days."""
    other_cost_per_dollar_bps: float = 0.70
    """Commission and regulatory fees per dollar traded -- unavoidable whether
    the order is passive or aggressive."""
    fixed_cost_bps: float = 25.0
    """Borrow. Rent on the short leg; does not scale with turnover at all."""
    daily_book_sharpe: float = 0.84
    """The incumbent. Intraday has to beat this, not merely beat zero."""

    def turnover(self, fraction_per_bar: float) -> float:
        """Annualized one-way turnover as a multiple of NAV."""
        return self.bars_per_year * fraction_per_bar


def net_sharpe(
    fill_ratio: float,
    adverse_selection: float,
    fraction_per_bar: float,
    economics: ExecutionEconomics | None = None,
) -> float:
    """Net Sharpe under passive execution, in the model described above.

    Built from four annualized terms, all in basis points of NAV:

        gross            = IR x book volatility
        spread edge      = turnover x half-spread x (2p - 1)
        adverse selection= -turnover x p x a x per-bar volatility
        other costs      = -turnover x per-dollar fees  -  borrow

    The ``(2p - 1)`` factor is the whole thesis in one coefficient: below a 50%
    fill ratio we are still net payers of the spread, above it we are net
    earners, and the sign flip is what could make high turnover affordable.
    """
    e = economics or ExecutionEconomics()
    turnover = e.turnover(fraction_per_bar)

    gross = e.implied_gross_ir * e.book_vol_bps
    spread_edge = turnover * e.half_spread_bps * (2.0 * fill_ratio - 1.0)
    adverse = -turnover * fill_ratio * adverse_selection * e.bar_vol_bps
    other = -turnover * e.other_cost_per_dollar_bps - e.fixed_cost_bps

    return float((gross + spread_edge + adverse + other) / e.book_vol_bps)


def feasibility_map(
    fill_ratios: np.ndarray | list[float],
    adverse_levels: np.ndarray | list[float],
    fraction_per_bar: float = 0.10,
    economics: ExecutionEconomics | None = None,
) -> pd.DataFrame:
    """Net Sharpe over the (fill ratio, adverse selection) grid.

    Rows are adverse-selection levels, columns fill ratios -- so reading left to
    right shows better fills helping and reading top to bottom shows adverse
    selection taking it back.
    """
    e = economics or ExecutionEconomics()
    grid = pd.DataFrame(
        [[net_sharpe(p, a, fraction_per_bar, e) for p in fill_ratios] for a in adverse_levels],
        index=pd.Index([round(float(a), 4) for a in adverse_levels], name="adverse_selection"),
        columns=pd.Index([round(float(p), 4) for p in fill_ratios], name="fill_ratio"),
    )
    return grid


def required_fill_ratio(
    adverse_selection: float,
    fraction_per_bar: float = 0.10,
    economics: ExecutionEconomics | None = None,
) -> float:
    """Smallest fill ratio that beats the daily book at a given adverse selection.

    Solves the linear model for ``p``. Returns NaN when no fill ratio clears the
    bar -- which is the genuinely useful answer, because it says the idea fails
    at that adverse-selection level however well the orders are placed.
    """
    e = economics or ExecutionEconomics()
    turnover = e.turnover(fraction_per_bar)

    # net = (gross - turnover*fees - borrow - turnover*hs + p*turnover*(2*hs - a*bar_vol)) / vol
    target = e.daily_book_sharpe * e.book_vol_bps
    constant = (
        e.implied_gross_ir * e.book_vol_bps
        - turnover * e.other_cost_per_dollar_bps
        - e.fixed_cost_bps
        - turnover * e.half_spread_bps
    )
    slope = turnover * (2.0 * e.half_spread_bps - adverse_selection * e.bar_vol_bps)
    if slope <= 0:
        # Filling more makes things worse: adverse selection exceeds the spread
        # earned, so passive execution cannot help at this level.
        return float("nan")
    needed = (target - constant) / slope
    return float(needed) if 0.0 <= needed <= 1.0 else float("nan")


def break_even_adverse_selection(
    fill_ratio: float,
    fraction_per_bar: float = 0.10,
    economics: ExecutionEconomics | None = None,
) -> float:
    """Most adverse selection tolerable at a given fill ratio, in units of bar vol.

    The number to take to a microstructure study: it converts "is passive
    execution viable" into a single measurable quantity.
    """
    e = economics or ExecutionEconomics()
    turnover = e.turnover(fraction_per_bar)
    if fill_ratio <= 0:
        return float("nan")

    target = e.daily_book_sharpe * e.book_vol_bps
    without_adverse = (
        e.implied_gross_ir * e.book_vol_bps
        + turnover * e.half_spread_bps * (2.0 * fill_ratio - 1.0)
        - turnover * e.other_cost_per_dollar_bps
        - e.fixed_cost_bps
    )
    slack = without_adverse - target
    return float(slack / (turnover * fill_ratio * e.bar_vol_bps))
