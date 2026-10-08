"""The full study: run the strategy, then run everything that tries to disprove it.

Order matters. The baseline backtest comes first, then the Fundamental Law
diagnostic explains it, then the Section 10 controls attack it. The trial count
that feeds the Deflated Sharpe Ratio is taken from the run cache rather than
guessed, so every configuration this study touches is counted against itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import validation as v
from .config import Config
from .costs.model import attribution_bps
from .data.loader import Dataset, load_dataset
from .law import FundamentalLawReport
from .law import analyze as analyze_law
from .reporting.metrics import sharpe, summarize
from .risk.constraints import binding_constraint_report
from .risk.neutralize import neutrality_report
from .strategy import StrategyRun, activity_table, run_strategy

log = logging.getLogger(__name__)

TARGET_SHARPE = 2.0
TRADE_BAND = (5, 50)


@dataclass
class Study:
    cfg: Config
    run: StrategyRun
    law: FundamentalLawReport
    performance: pd.DataFrame
    activity: pd.Series
    attribution: pd.Series
    neutrality: pd.DataFrame
    constraints: pd.DataFrame
    walk_forward: v.WalkForwardResult
    sensitivity: pd.DataFrame
    surfaces: dict[str, pd.DataFrame]
    plateau: pd.Series
    cost_sweep: pd.DataFrame
    break_even: float
    cost_toggles: pd.DataFrame
    capacity: pd.DataFrame
    bootstrap: pd.Series
    bootstrap_draws: np.ndarray
    deflated: pd.Series
    pbo: pd.Series
    plots: list[Path] = field(default_factory=list)

    @property
    def oos_net_sharpe(self) -> float:
        return sharpe(self.walk_forward.oos_returns)

    @property
    def clears_target(self) -> bool:
        return bool(np.isfinite(self.oos_net_sharpe) and self.oos_net_sharpe >= TARGET_SHARPE)


def run_study(
    cfg: Config,
    outdir: Path | str = "output",
    grid: dict[str, Any] | None = None,
    dataset: Dataset | None = None,
    make_plots: bool = True,
    prior_trials: int = 0,
) -> Study:
    """Run the baseline and every Section 10 control.

    ``prior_trials`` is the number of configurations already examined *before*
    this study — exploratory sweeps, a turnover-tuning pass, an earlier run whose
    result informed these defaults. They are added to this study's own grid size
    when deflating the Sharpe. Spec S10 asks for the trial count to be tracked
    honestly and fed in, and a config that was chosen by looking at results is a
    trial whether or not it was run inside this function.
    """
    outdir = Path(outdir)
    dataset = dataset if dataset is not None else load_dataset(cfg)

    log.info("baseline backtest")
    run = run_strategy(cfg, dataset)

    log.info("fundamental law diagnostic")
    law = analyze_law(run, TARGET_SHARPE)

    log.info("parameter sweep and walk-forward")
    grid = grid if grid is not None else v.DEFAULT_GRID
    sensitivity, cache = v.sweep(cfg, dataset, grid)
    walk_forward = v.walk_forward(cfg, dataset, grid, cache)

    surfaces = _build_surfaces(sensitivity, grid)
    plateau = v.plateau_score(
        next(iter(surfaces.values())) if surfaces else sensitivity["net_sharpe"].to_numpy()
    )

    log.info("cost sensitivity and capacity")
    cost_sweep = v.sweep_cost_multiplier(run)
    break_even = v.break_even_multiplier(cost_sweep)
    cost_toggles = v.component_toggle_table(run)
    capacity = v.capacity_sweep(
        run, [cfg.execution.initial_nav * m for m in (0.1, 0.5, 1, 5, 20, 100)]
    )

    log.info("bootstrap, deflated sharpe and PBO")
    oos = walk_forward.oos_returns
    vc = cfg.validation
    bootstrap = v.block_bootstrap_sharpe(oos, vc.bootstrap_samples, vc.bootstrap_block, seed=vc.seed)
    draws = v.bootstrap_distribution(oos, vc.bootstrap_samples, vc.bootstrap_block, seed=vc.seed)

    all_returns = cache.returns()
    trial_sharpes = all_returns.apply(lambda s: s.mean() / s.std(ddof=1) if s.std(ddof=1) else np.nan)
    deflated = v.deflated_sharpe(oos, cache.n_trials + prior_trials, trial_sharpes)
    pbo = v.probability_of_backtest_overfitting(all_returns, vc.pbo_splits)

    performance = _performance_table(run, walk_forward)
    study = Study(
        cfg=cfg,
        run=run,
        law=law,
        performance=performance,
        activity=activity_table(run.result),
        attribution=attribution_bps(run.result.costs, run.result.nav),
        neutrality=neutrality_report(run.result.weights, run.portfolio.beta, dataset.prices.sectors),
        constraints=binding_constraint_report(
            run.result.weights,
            dataset.prices.adv(cfg.data.adv_lookback),
            cfg.execution.initial_nav,
            cfg.risk,
        ),
        walk_forward=walk_forward,
        sensitivity=sensitivity,
        surfaces=surfaces,
        plateau=plateau,
        cost_sweep=cost_sweep,
        break_even=break_even,
        cost_toggles=cost_toggles,
        capacity=capacity,
        bootstrap=bootstrap,
        bootstrap_draws=draws,
        deflated=deflated,
        pbo=pbo,
    )

    if make_plots:
        study.plots = _make_plots(study, outdir)
    return study


def _performance_table(run: StrategyRun, wf: v.WalkForwardResult) -> pd.DataFrame:
    """Gross/net x IS/OOS, the four numbers the spec insists on seeing together."""
    result = run.result
    oos_index = wf.oos_returns.index
    is_mask = ~result.net_return.index.isin(oos_index)
    return pd.concat(
        [
            summarize(result.gross_return, "full_gross"),
            summarize(result.net_return, "full_net"),
            summarize(result.gross_return[is_mask], "IS_gross"),
            summarize(result.net_return[is_mask], "IS_net"),
            summarize(result.gross_return.reindex(oos_index), "OOS_gross"),
            summarize(wf.oos_returns, "OOS_net_selected"),
            summarize(wf.oos_returns_default, "OOS_net_default"),
        ],
        axis=1,
    )


def _build_surfaces(sensitivity: pd.DataFrame, grid: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Pivot every pair of swept parameters into a heatmap-shaped frame."""
    keys = [k for k in grid if k in sensitivity.columns and sensitivity[k].nunique() > 1]
    surfaces: dict[str, pd.DataFrame] = {}
    for i, key_a in enumerate(keys):
        for key_b in keys[i + 1 :]:
            try:
                surfaces[f"{key_a}__vs__{key_b}"] = v.surface(sensitivity, key_a, key_b)
            except Exception:  # non-rectangular grid slice
                continue
    if not surfaces and keys:
        key = keys[0]
        surfaces[key] = sensitivity.set_index(key)[["net_sharpe"]]
    return surfaces


def _make_plots(study: Study, outdir: Path) -> list[Path]:
    from .reporting import plots as P

    paths = P.all_plots(study.run, outdir, extras={"ic_series": study.law.ic_series})
    paths.append(P.cost_sensitivity_plot(study.cost_sweep, outdir, study.break_even))
    if not study.walk_forward.folds.empty:
        paths.append(P.walk_forward_plot(study.walk_forward.folds, outdir))
    if len(study.bootstrap_draws) and not study.bootstrap.empty:
        paths.append(
            P.bootstrap_plot(
                study.bootstrap_draws,
                float(study.bootstrap["sharpe"]),
                (float(study.bootstrap["ci_low"]), float(study.bootstrap["ci_high"])),
                outdir,
            )
        )
    for i, (name, frame) in enumerate(study.surfaces.items()):
        if frame.shape[1] > 1:
            paths.append(P.sensitivity_heatmap(frame, outdir, name=f"sensitivity_{i}"))
    return paths
