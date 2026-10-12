"""Phase 2, direction B: how good would passive fills have to be?

Pure arithmetic over the measured cost structure -- no market data. It killed
the direction in one line: commission and fees per dollar traded exceed the most
spread a perfect limit order could ever capture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..execution import (ExecutionEconomics, break_even_adverse_selection, feasibility_map,
                         required_fill_ratio)

DESCRIPTION = "Passive-execution feasibility: fill ratio x adverse selection"
FILLS = np.arange(0.3, 1.01, 0.1)
ADVERSE = np.array([0.0, 0.002, 0.005, 0.01, 0.02, 0.05])
FRACTIONS = (0.02, 0.05, 0.10, 0.20)


def run() -> dict[str, pd.DataFrame]:
    e = ExecutionEconomics()
    out = {f"phase2_feasibility_{int(f * 100)}": feasibility_map(FILLS, ADVERSE, f, e) for f in (0.05, 0.10)}
    required = pd.DataFrame([
        {"frac_per_bar": f, "adverse_selection": a, "required_fill": required_fill_ratio(a, f, e)}
        for f in FRACTIONS for a in ADVERSE
    ]).pivot(index="adverse_selection", columns="frac_per_bar", values="required_fill")
    tolerable = pd.DataFrame([
        {"frac_per_bar": f, **{f"p={p:.1f}": break_even_adverse_selection(p, f, e)
                               for p in (0.5, 0.6, 0.7, 0.8, 0.9)}}
        for f in FRACTIONS
    ]).set_index("frac_per_bar")
    economics = pd.DataFrame([{
        "commission_fees_bps_per_dollar": e.other_cost_per_dollar_bps,
        "half_spread_bps": e.half_spread_bps, "implied_gross_ir": e.implied_gross_ir,
        "daily_book_sharpe": e.daily_book_sharpe,
    }])
    return {**out, "phase2_required_fill": required, "phase2_tolerable_adverse": tolerable,
            "phase2_economics": economics}
