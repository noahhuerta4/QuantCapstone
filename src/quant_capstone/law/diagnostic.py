"""Implied vs realized IR: the Fundamental Law used as a measuring instrument
rather than as decoration (spec S7).

For a dollar-neutral book the benchmark is zero, so the information ratio and the
Sharpe ratio are the same number. That is what makes this diagnostic bite: we can
put ``IR = TC * IC * sqrt(BR)`` next to the Sharpe the backtest actually produced
and ask why they differ.

Reading the output:

* **Implied and realized agree.** The law is describing the strategy correctly,
  and the terms tell you what to attack.
* **Implied > realized.** The bets are less independent than even the corrected
  breadth says, or the IC we measured is not the IC the portfolio expressed
  (usually: the weights are driven by risk sizing more than by alpha).
* **Implied < realized.** Be suspicious before being pleased. The usual causes
  are a look-ahead leak or an unmodelled cost, not a discovery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.engine import forward_returns
from ..config import TRADING_DAYS
from ..reporting.metrics import sharpe
from .breadth import (
    breadth_implied_by_performance,
    effective_breadth,
    required_ic,
    seductive_lie_table,
)
from .ic import ic_decay, ic_summary, information_coefficient
from .transfer import transfer_summary


@dataclass
class FundamentalLawReport:
    ic_series: pd.Series
    ic: pd.Series
    ic_residual: pd.Series
    """IC against *residual* forward returns -- the ones a neutral book trades."""
    ic_decay: pd.Series
    breadth: pd.Series
    transfer: pd.Series
    comparison: pd.Series
    seductive_lie: pd.DataFrame
    verdict: str

    def to_frame(self) -> pd.DataFrame:
        blocks = [self.ic, self.ic_residual, self.breadth, self.transfer, self.comparison]
        return pd.concat(blocks).to_frame("value")


def analyze(run, target_ir: float = 2.0, n_factors_removed: int = 1) -> FundamentalLawReport:
    """Full Section 7 diagnostic for a completed :class:`StrategyRun`."""
    prices = run.dataset.prices
    fwd = forward_returns(prices, run.cfg.execution.timing)

    ic_series = information_coefficient(run.signal, fwd)
    ic_stats = ic_summary(ic_series)
    decay = ic_decay(run.signal, prices.returns(), horizons=[1, 2, 3, 5, 10, 21])

    # The book is beta- and sector-neutral, so the return it is actually trying to
    # forecast is the residual, not the raw forward return. Scoring the signal
    # against the raw return mixes in a factor bet the strategy is not allowed to
    # take, and is the more apples-to-apples number for the law's IC term.
    from ..factors import sector_dummies
    from ..signals.orthogonalize import orthogonalize

    fwd_residual = orthogonalize(
        fwd,
        beta=run.portfolio.beta,
        sector_loadings=sector_dummies(prices.sectors, prices.tickers),
    )
    ic_residual_stats = ic_summary(information_coefficient(run.signal, fwd_residual))
    ic_residual_stats.index = [f"residual_{i}" for i in ic_residual_stats.index]

    breadth = effective_breadth(
        prices.returns(), run.signal, run.dataset.members, n_factors_removed
    )

    gross_sharpe = sharpe(run.result.gross_return)
    net_sharpe = sharpe(run.result.net_return)
    transfer = transfer_summary(
        run.portfolio.ideal_weights, run.portfolio.weights, gross_sharpe, net_sharpe
    )

    ic_mean = ic_stats.get("ic_mean", np.nan)
    br_eff = breadth["breadth_effective"]
    br_naive = breadth["breadth_naive"]

    comparison = pd.Series(
        {
            "implied_ir_naive": ic_mean * np.sqrt(br_naive),
            "implied_ir_effective_gross": transfer["tc_constraints"] * ic_mean * np.sqrt(br_eff),
            "implied_ir_effective_net": transfer["tc_total"] * ic_mean * np.sqrt(br_eff),
            "realized_sharpe_gross": gross_sharpe,
            "realized_sharpe_net": net_sharpe,
            "required_ic_for_target": required_ic(target_ir, br_eff, transfer["tc_total"]),
            "actual_ic": ic_mean,
            "ic_shortfall_multiple": (
                required_ic(target_ir, br_eff, transfer["tc_total"]) / ic_mean
                if ic_mean and np.isfinite(ic_mean) and ic_mean != 0
                else np.nan
            ),
        }
    )
    comparison["implied_minus_realized_net"] = (
        comparison["implied_ir_effective_net"] - net_sharpe
    )
    # Breadth read off the realized result rather than counted a priori. The
    # ratio to breadth_effective is the cleanest statement of how much our bet
    # counting is off by, and in which direction.
    comparison["breadth_implied_by_realized"] = breadth_implied_by_performance(
        gross_sharpe, ic_mean, transfer["tc_constraints"]
    )
    comparison["breadth_effective_over_implied"] = (
        br_eff / comparison["breadth_implied_by_realized"]
        if np.isfinite(comparison["breadth_implied_by_realized"])
        and comparison["breadth_implied_by_realized"] > 0
        else np.nan
    )

    lie = seductive_lie_table(breadth, target_ir, transfer["tc_total"])
    return FundamentalLawReport(
        ic_series=ic_series,
        ic=ic_stats,
        ic_residual=ic_residual_stats,
        ic_decay=decay,
        breadth=breadth,
        transfer=transfer,
        comparison=comparison,
        seductive_lie=lie,
        verdict=_verdict(comparison, transfer, breadth, target_ir),
    )


def _verdict(comparison: pd.Series, transfer: pd.Series, breadth: pd.Series, target_ir: float) -> str:
    """Name the binding constraint in plain words. This is the Section 1.6 answer."""
    net = comparison["realized_sharpe_net"]
    gross = comparison["realized_sharpe_gross"]
    tc_costs = transfer["tc_costs"]
    tc_constraints = transfer["tc_constraints"]
    shortfall = comparison["ic_shortfall_multiple"]

    # This diagnostic runs on the full sample, so it must not claim to be
    # quoting an out-of-sample figure -- the walk-forward owns that number.
    lines = []
    verb = "clears" if np.isfinite(net) and net >= target_ir else "falls short of"
    lines.append(f"Full-sample net Sharpe of {net:.2f} {verb} the {target_ir:.1f} target.")

    if np.isfinite(tc_costs) and tc_costs < 0.5:
        lines.append(
            f"Costs are the dominant constraint: they cut the Sharpe from {gross:.2f} gross to "
            f"{net:.2f} net, retaining only {tc_costs:.0%} of it. Improving the forecast will not "
            f"fix this -- every increment of IC is taxed at the same rate."
        )
    elif np.isfinite(shortfall) and shortfall > 1.2:
        lines.append(
            f"Forecasting skill is the binding constraint: hitting IR {target_ir:.1f} at the "
            f"measured breadth and transfer coefficient needs an IC of "
            f"{comparison['required_ic_for_target']:.4f}, and the signal delivers "
            f"{comparison['actual_ic']:.4f} -- short by a factor of {shortfall:.1f}."
        )
    else:
        lines.append(
            "Neither term is clearly binding on its own; costs and forecasting skill are "
            "constraining the result to a similar degree."
        )

    lines.append(
        f"Breadth is the quiet killer: the naive count of {breadth['breadth_naive']:,.0f} bets per "
        f"year is really about {breadth['breadth_effective']:,.0f} once cross-sectional dependence "
        f"({breadth['n_eff_after_factor_removal']:.1f} independent directions, not "
        f"{breadth['mean_names']:.0f} names) and signal persistence "
        f"(lag-1 autocorrelation {breadth['signal_autocorr_lag1']:.2f}) are accounted for -- a "
        f"{1 / breadth['breadth_shrinkage']:,.0f}x reduction, which raises the required IC by "
        f"{np.sqrt(1 / breadth['breadth_shrinkage']):,.0f}x."
    )
    if np.isfinite(tc_constraints):
        lines.append(
            f"Portfolio constraints retain {tc_constraints:.0%} of the ideal alpha "
            f"(transfer coefficient before costs)."
        )
    return " ".join(lines)
