"""Point-in-time universe construction (spec S3).

Membership on day t is decided using data strictly before t. ADV is already
shifted by one bar in ``PriceData.adv``, and eligibility screens use lagged
prices, so nothing here can see the day it is selecting for.

We re-rank only every ``universe_refresh`` bars and hold membership constant in
between. That is both realistic (you do not re-shuffle your tradable list daily)
and important for turnover: daily re-ranking would churn the book at the
universe boundary for no alpha reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DataConfig
from .base import PriceData


def build_universe(data: PriceData, cfg: DataConfig) -> pd.DataFrame:
    """Return a boolean ``date x ticker`` membership matrix."""
    adv = data.adv(cfg.adv_lookback)
    close_prev = data.close_raw.shift(1)
    history = data.close.notna().cumsum().shift(1)

    eligible = (
        adv.notna()
        & (adv > 0)
        & (close_prev >= cfg.min_price)
        & (history >= cfg.min_history)
        & data.close.notna()          # must have a tradable print today
        & data.open_raw.notna()
    )

    scores = adv.where(eligible)
    refresh_rows = np.arange(len(data.dates)) % max(cfg.universe_refresh, 1) == 0

    members = pd.DataFrame(False, index=data.dates, columns=data.tickers)
    selected: pd.Index = pd.Index([], dtype=object)
    for i, date in enumerate(data.dates):
        if refresh_rows[i]:
            row = scores.iloc[i].dropna()
            if not row.empty:
                selected = row.nlargest(cfg.universe_size).index
        if len(selected):
            # Keep the held list, but drop any name that is untradable today.
            members.loc[date, selected] = True
    members &= eligible

    return members


def universe_stats(members: pd.DataFrame) -> pd.DataFrame:
    """Daily count plus turnover of the membership list itself."""
    count = members.sum(axis=1)
    entering = (members & ~members.shift(1, fill_value=False)).sum(axis=1)
    leaving = (~members & members.shift(1, fill_value=False)).sum(axis=1)
    return pd.DataFrame({"n_members": count, "entering": entering, "leaving": leaving})
