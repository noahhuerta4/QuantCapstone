"""Two further sleeves off the earnings clock, reusing the XBRL panel.

The Phase-1 result worth building on: an earnings *surprise* earns its place in
the blend where a static ratio does not, because the surprise is an event with
its own clock and the ratio is a level that sits still. If that is the mechanism,
then the productive question is not "what other ratios exist" but "what other
signals ride the same clock".

Two do, and they are close to orthogonal to the existing PEAD sleeve by
construction:

``revenue_surprise``
    The same seasonal-random-walk surprise, measured on revenue instead of
    earnings. Revenue is harder to manage than the bottom line — fewer accounting
    levers sit between the customer and the top line — so a revenue surprise says
    something an EPS surprise does not.

``announcement_reversal``
    The *reversal* of the price move around the report, where PEAD is the
    *drift*. Same event, opposite sign, different horizon: overreaction on the
    day unwinds over the next few sessions while the informational content drifts
    for weeks. Two signals from one event that are near-orthogonal because they
    describe different parts of it.

**A caveat that belongs in front, not in a footnote.** We anchor on the XBRL
filing date because that is the date we can prove was public. The earnings press
release (8-K Item 2.02) usually lands a few days *earlier*, so the filing date is
a lagged proxy for the true announcement. That is conservative — it can only make
the measured reversal weaker, never spuriously stronger — but it does mean this
sleeve is being tested with one hand tied. Recovering the true announcement date
needs 8-K item parsing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fundamental import standardized_unexpected_earnings


def revenue_surprise(
    quarterly_revenue: pd.DataFrame, decay_days: int = 60
) -> pd.DataFrame:
    """Standardized unexpected revenue, on the same event clock as PEAD."""
    return standardized_unexpected_earnings(quarterly_revenue, decay_days=decay_days)


def earnings_event_dates(quarterly_income: pd.DataFrame) -> pd.DataFrame:
    """Boolean panel marking the row where each name's reported quarter changes.

    The panel arrives forward-filled from the filing date, so a change in value
    *is* the arrival of a new report.
    """
    return quarterly_income.notna() & quarterly_income.ne(quarterly_income.shift(1))


def announcement_reversal(
    close: pd.DataFrame,
    quarterly_income: pd.DataFrame,
    reaction_window: int = 3,
    decay_days: int = 5,
) -> pd.DataFrame:
    """Fade the price move around an earnings report (Bernard & Thomas 1989).

    On the day a report becomes public, score each name by the negative of its
    recent return, and fade that score to zero over the next few sessions. The
    bet is that the immediate reaction overshoots.

    Short by design: ``decay_days`` defaults to 5 against PEAD's 60. The
    overreaction unwinds in days while the information drifts for a quarter, and
    giving them the same horizon would collapse two distinct effects into one.
    """
    events = earnings_event_dates(quarterly_income)
    reaction = close.pct_change(reaction_window)

    values = np.full(close.shape, np.nan)
    event_matrix = events.to_numpy()
    reaction_matrix = reaction.reindex_like(close).to_numpy(dtype=float)
    n_days = len(close)

    for column in range(close.shape[1]):
        for row in np.flatnonzero(event_matrix[:, column]):
            move = reaction_matrix[row, column]
            if not np.isfinite(move):
                continue
            stop = min(row + decay_days, n_days)
            fade = 1.0 - np.arange(stop - row) / decay_days
            values[row:stop, column] = -move * fade

    return pd.DataFrame(values, index=close.index, columns=close.columns)


def event_density(events: pd.DataFrame, members: pd.DataFrame) -> pd.Series:
    """How often the event clock actually ticks across the universe.

    The number that decides whether an event sleeve can have breadth at all: if
    reports cluster into four narrow windows a year, the cross-section is static
    between them however well-targeted the signal is.
    """
    in_universe = events.where(members.reindex_like(events).fillna(False))
    per_day = in_universe.sum(axis=1)
    return pd.Series(
        {
            "mean_events_per_day": float(per_day.mean()),
            "median_events_per_day": float(per_day.median()),
            "max_events_per_day": float(per_day.max()),
            "days_with_any_event": float((per_day > 0).mean()),
            "total_events": float(per_day.sum()),
        }
    )
