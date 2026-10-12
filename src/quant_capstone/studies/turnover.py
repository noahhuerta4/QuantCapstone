"""First turnover tuning: rebalance interval x no-trade band x universe size.

The earliest sweep in the project, run before the blend, before calibrated
spreads and before ``k`` moved to 2. It is exploratory -- nothing here was
adopted directly -- and is kept because it is where "rebalance every day costs
more than the edge is worth" was first measured.
"""

from __future__ import annotations

import pandas as pd

from .. import validation as v
from ..data import load_dataset
from . import eras

DESCRIPTION = "Rebalance x band x universe on the single reversal sleeve (exploratory)"
GRID = {
    "execution__no_trade_band": [0.003, 0.008],
    "execution__rebalance_every": [1, 3, 5, 10],
    "data__universe_size": [50, 100, 200],
}


def run() -> dict[str, pd.DataFrame]:
    cfg = eras.FLAT_SPREAD.override(signal__k=5, data__universe_size=200)
    table, _ = v.sweep(cfg, load_dataset(cfg), GRID)
    cols = ["data__universe_size", "execution__rebalance_every", "execution__no_trade_band",
            "trades_per_day", "ann_turnover", "annual_cost_bps", "gross_sharpe", "net_sharpe"]
    table = table[cols].sort_values("net_sharpe", ascending=False)
    table["in_band"] = table["trades_per_day"].between(5, 50)
    return {"turnover_tuning": table}
