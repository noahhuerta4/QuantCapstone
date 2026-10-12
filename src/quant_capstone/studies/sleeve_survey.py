"""Every sleeve built in Phase 1 on one chart: IC against persistence.

The figure that explains which sleeves make the blend: forecasting skill only
becomes information ratio if the signal turns over. The three sleeves the book
actually uses are also backtested standalone under the final configuration, so
the README's sleeve table comes from one place and one cost model.
"""

from __future__ import annotations

import pandas as pd

from ..backtest import forward_returns
from ..data import load_dataset
from ..law import information_coefficient
from ..reporting.metrics import sharpe
from ..signals import build_sleeves
from . import eras
from .common import autocorr, backtest, t_stat, turnover

DESCRIPTION = "IC and autocorrelation for all fourteen Phase 1 sleeves"
ALL = ("volume_shock_reversal", "overnight_reversal", "reversal", "momentum",
       "residual_momentum", "low_volatility", "idiosyncratic_volatility",
       "earnings_surprise", "book_to_price", "earnings_yield", "gross_profitability",
       "return_on_equity", "asset_growth", "accruals")
USED = ("volume_shock_reversal", "overnight_reversal", "earnings_surprise")


def run() -> dict[str, pd.DataFrame]:
    cfg = eras.PEAD.override(signal__alphas=ALL, signal__pead_weight=0.0)
    ds = load_dataset(cfg)
    fwd = forward_returns(ds.prices, "next_open")
    rows = []
    for name, sleeve in build_sleeves(ds, cfg).items():
        ic = information_coefficient(sleeve, fwd).dropna()
        rows.append({"sleeve": name, "ic_mean": float(ic.mean()), "ic_t": t_stat(ic, ddof=1),
                     "autocorr": autocorr(sleeve), "used": name in USED})
    survey = pd.DataFrame(rows)

    final = eras.FINAL
    ds_final = load_dataset(final)
    standalone = []
    for name in USED:
        r = backtest(ds_final, final.override(signal__alphas=(name,), signal__pead_weight=0.0))
        standalone.append({"sleeve": name, "gross": sharpe(r.gross_return),
                           "net": sharpe(r.net_return), "turnover": turnover(r)})
    used = survey.set_index("sleeve").loc[list(USED), ["ic_mean", "ic_t", "autocorr"]].reset_index()
    return {"all_sleeves": survey, "used_sleeves": used.merge(pd.DataFrame(standalone), on="sleeve")}
