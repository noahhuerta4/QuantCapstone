"""Every figure in the write-up, regenerated from the final configuration.

Writes PNGs to ``Report/figures`` (version-controlled, so the write-up builds
from a clean checkout). Five figures are drawn from other studies' CSVs, so
this study runs after them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import validation as v
from ..backtest import run_backtest
from ..costs import attribution_bps
from ..data import load_dataset
from ..law import analyze
from ..reporting import plots as P
from ..risk.pipeline import build_portfolio
from ..signals import build_signal, build_sleeves, sleeve_correlations
from ..strategy import run_strategy
from . import eras
from .bayes import HEADLINE
from .sleeve_survey import ALL

DESCRIPTION = "All write-up figures (to Report/figures)"
FIGURES = Path("Report/figures")
NEEDS = ("liquidity", "pead", "sleeve_survey", "bayes")


def _portfolio_evolution(out: Path) -> Path:
    nav0 = 1_000_000.0
    cfg = eras.FINAL.override(execution__initial_nav=nav0)
    ds = load_dataset(cfg)
    r = run_backtest(ds, build_portfolio(ds, build_signal(ds, cfg), cfg).weights, cfg)
    nav = r.nav.copy()
    nav.iloc[-1] = float(r.nav.iloc[-1] * (1 + r.net_return.iloc[-1]))
    return P.portfolio_evolution({"net of costs": nav}, out, start_value=nav0)


def _selected_rule(studies_dir: Path, out: Path) -> None:
    """Walk-forward folds and bootstrap for the headline selection rule of the bayes study."""
    sel = pd.read_csv(studies_dir / "bayes_selections.csv", parse_dates=["test_start"])
    sel = sel[sel.method == HEADLINE].reset_index(drop=True)
    oos = pd.read_csv(studies_dir / "bayes_oos_returns.csv", index_col=0, parse_dates=True)[HEADLINE]

    starts = list(sel.test_start)
    ends = [oos.index[oos.index < s][-1] for s in starts[1:]] + [oos.index[-1]]
    sel["fold"] = [f"{s.date()}..{e.date()}" for s, e in zip(starts, ends)]
    P.walk_forward_plot(sel, out, name="walk_forward_selected")

    vc = eras.FINAL.validation
    bs = v.block_bootstrap_sharpe(oos, vc.bootstrap_samples, vc.bootstrap_block, seed=vc.seed)
    draws = v.bootstrap_distribution(oos, vc.bootstrap_samples, vc.bootstrap_block, seed=vc.seed)
    P.bootstrap_plot(draws, float(bs["sharpe"]), (float(bs["ci_low"]), float(bs["ci_high"])), out,
                     name="bootstrap_sharpe_selected")


def run(studies_dir: Path = Path("results/studies"), out: Path = FIGURES) -> dict[str, pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    cfg = eras.FINAL
    ds = load_dataset(cfg)
    strat = run_strategy(cfg, ds)
    r, law = strat.result, analyze(strat)

    P.equity_and_drawdown(r.gross_return, r.net_return, out)
    P.rolling_sharpe_plot(r.net_return, out)
    P.cost_attribution(attribution_bps(r.costs, r.nav), out)
    P.exposures(r.exposures, out)
    P.trades_per_day(r.n_trades, out)
    P.ic_plot(law.ic_series, out)
    P.ic_decay_plot(law.ic_decay, out)
    P.breadth_waterfall(law.breadth, out)
    _portfolio_evolution(out)

    P.sleeve_scatter(pd.read_csv(studies_dir / "all_sleeves.csv"), out)
    sleeves = build_sleeves(ds, cfg.override(signal__alphas=ALL, signal__pead_weight=0.0))
    P.sleeve_correlation_heatmap(sleeve_correlations(sleeves), out)
    P.liquidity_deciles(pd.read_csv(studies_dir / "liquidity_deciles.csv", index_col=0), out)
    P.weight_sweep(pd.read_csv(studies_dir / "pead_weight_sweep.csv"), out, "pead_weight", chosen=0.10)
    _selected_rule(studies_dir, out)

    sweep = v.sweep_cost_multiplier(strat)
    P.cost_sensitivity_plot(sweep, out, v.break_even_multiplier(sweep))
    vc = cfg.validation
    wf = v.walk_forward(cfg, ds, v.DEFAULT_GRID)
    P.walk_forward_plot(wf.folds, out)
    bs = v.block_bootstrap_sharpe(wf.oos_returns, vc.bootstrap_samples, vc.bootstrap_block, seed=vc.seed)
    draws = v.bootstrap_distribution(wf.oos_returns, vc.bootstrap_samples, vc.bootstrap_block, seed=vc.seed)
    P.bootstrap_plot(draws, float(bs["sharpe"]), (float(bs["ci_low"]), float(bs["ci_high"])), out)
    return {}
