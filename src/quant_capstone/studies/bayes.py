"""Does finer per-fold tuning help? Coarse grid vs fine grid vs Bayesian optimization.

The delivered walk-forward picks the best of 27 coarse cells per fold, and already
does worse out of sample than simply keeping the defaults. This study asks whether
searching harder changes that. Three search spaces, each selected two ways:

* **coarse**: the 27-cell production grid (PEAD weight x rebalance x band).
* **fine**: 175 cells over the same three parameters at finer spacing.
* **bayes**: Gaussian-process Bayesian optimization over four continuous
  parameters -- PEAD weight, the volume-shock share of the reversal weight,
  rebalance interval and no-trade band.

**argmax** takes the configuration with the best training net Sharpe.
**neighbourhood** takes the one whose *neighbourhood* has the best mean training
Sharpe: adjacent cells for the grids, the nearest evaluated points for the
optimizer. It trades peak height for flatness, the robust-selection idea.

Every candidate is run once over the full history and sliced per fold. That is
legitimate for the same reason it is in ``validation/walkforward.py``: the
pipeline is causal, so a configuration's day-t weights do not depend on anything
after t. The optimizer is warm-started: at each fold it rescores every point
evaluated in earlier folds on the new training window (free, the returns are
cached) and then adds new points. Selection still sees training data only.

All methods share one set of folds, embargoed by the longest rebalance interval
any of them can choose (15 days), so differences come from selection alone.
"""

from __future__ import annotations

import itertools
import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from scipy.stats import norm

from .. import validation as v
from ..data import load_dataset
from ..reporting.metrics import sharpe
from ..risk.pipeline import build_portfolio
from ..signals import build_signal
from ..backtest import run_backtest
from ..validation.walkforward import make_folds
from . import eras
from .common import fixed_blend

log = logging.getLogger(__name__)

DESCRIPTION = "Per-fold selection: coarse/fine grid vs GP Bayesian optimization, argmax vs neighbourhood"

VSR, OVN, PEAD = "volume_shock_reversal", "overnight_reversal", "earnings_surprise"

COARSE = {"pead": (0.0, 0.10, 0.20), "rebalance": (5, 10, 15), "band": (0.003, 0.006, 0.010)}
FINE = {"pead": (0.0, 0.05, 0.10, 0.15, 0.20), "rebalance": (5, 7, 10, 12, 15),
        "band": (0.002, 0.003, 0.004, 0.006, 0.008, 0.010, 0.012)}
DEFAULT = {"pead": 0.10, "vsr_share": 0.5, "rebalance": 10, "band": 0.006}

#: Optimizer search box. The band stops short of the 1.5% per-name cap, past which
#: nothing can ever trade and the Sharpe is undefined.
BOUNDS = {"pead": (0.0, 0.30), "vsr_share": (0.2, 0.8), "rebalance": (5, 15), "band": (0.0, 0.014)}
PARAMS = tuple(BOUNDS)
EMBARGO = 15
N_INIT = 12
N_PER_FOLD = 12
NEIGHBOURS = 9
"""Neighbourhood size for the optimizer, the point itself included."""
FAILED = -1.0
"""Score for a configuration with no defined Sharpe (it never trades)."""
HEADLINE = "fine neighbourhood interior"
"""The rule the write-up reports. Chosen after comparing all eight here, which is
why the diagnostics charge every configuration this study evaluated as a trial."""
PRIOR_TRIALS = 327
"""Configurations in the trial ledger before this study, as stated in the write-up."""


def _key(p: dict) -> tuple:
    """Round onto the resolution actually distinguishable, so the cache hits."""
    return (round(p["pead"], 2), round(p["vsr_share"], 2), int(round(p["rebalance"])),
            round(p["band"] * 2000) / 2000)


def _params(key: tuple) -> dict:
    return dict(zip(PARAMS, key))


@dataclass
class Evaluator:
    """Full-history net returns per configuration; portfolios shared across
    configurations that differ only in rebalance interval or band."""

    ds: object
    base: object
    portfolios: dict = field(default_factory=dict)
    returns: dict = field(default_factory=dict)

    def __call__(self, key: tuple) -> pd.Series:
        if key not in self.returns:
            pead, share, rebalance, band = key
            reversal = 1.0 - pead
            cfg = fixed_blend(self.base, {VSR: reversal * share, OVN: reversal * (1 - share), PEAD: pead})
            cfg = cfg.override(execution__rebalance_every=rebalance, execution__no_trade_band=band)
            signal_key = (pead, share)
            if signal_key not in self.portfolios:
                self.portfolios[signal_key] = build_portfolio(
                    self.ds, build_signal(self.ds, cfg), cfg).weights
            self.returns[key] = run_backtest(self.ds, self.portfolios[signal_key], cfg).net_return
        return self.returns[key]


def _score(series: pd.Series, lo, hi) -> float:
    value = sharpe(series.loc[lo:hi])
    return float(value) if np.isfinite(value) else FAILED


def _unit(key: tuple) -> np.ndarray:
    """Position of a configuration in the unit hypercube of the search box."""
    return np.array([(v - BOUNDS[p][0]) / (BOUNDS[p][1] - BOUNDS[p][0])
                     for p, v in zip(PARAMS, key)])


def _from_unit(u: np.ndarray) -> tuple:
    p = {name: BOUNDS[name][0] + x * (BOUNDS[name][1] - BOUNDS[name][0]) for name, x in zip(PARAMS, u)}
    return _key(p)


# ---------------------------------------------------------------- selection rules

def _grid_neighbourhood(scores: dict, grid: dict, interior: bool = False) -> dict:
    """Mean score over each cell and its neighbours (one step in every grid index).

    An edge cell's neighbourhood is truncated: it omits whatever lies beyond the
    edge, so it can outscore an interior cell by never seeing the bad side.
    ``interior=True`` keeps only cells whose neighbourhood is complete.
    """
    axes = [grid["pead"], grid["rebalance"], grid["band"]]
    index = {k: (axes[0].index(k[0]), axes[1].index(k[2]), axes[2].index(k[3])) for k in scores}
    by_index = {v: k for k, v in index.items()}
    sizes = [len(a) for a in axes]
    out = {}
    for k, (i, j, m) in index.items():
        if interior and not all(0 < x < n - 1 for x, n in zip((i, j, m), sizes)):
            continue
        nb = [by_index[(a, b, c)] for a in (i - 1, i, i + 1) for b in (j - 1, j, j + 1)
              for c in (m - 1, m, m + 1) if (a, b, c) in by_index]
        out[k] = float(np.mean([scores[n] for n in nb]))
    return out


def _knn_neighbourhood(scores: dict) -> dict:
    """Mean score over each point's nearest evaluated points in the unit box."""
    keys = list(scores)
    units = np.array([_unit(k) for k in keys])
    values = np.array([scores[k] for k in keys])
    k = min(NEIGHBOURS, len(keys))
    out = {}
    for i, key in enumerate(keys):
        nearest = np.argsort(np.linalg.norm(units - units[i], axis=1))[:k]
        out[key] = float(values[nearest].mean())
    return out


def _best(scores: dict):
    return max(scores, key=scores.get)


# ---------------------------------------------------------------- optimizer

def _propose(scores: dict, rng: np.random.Generator, n: int) -> list[tuple]:
    """n new configurations by expected improvement under a GP fitted to ``scores``."""
    keys = list(scores)
    X = np.array([_unit(k) for k in keys])
    y = np.array([scores[k] for k in keys])
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * Matern(length_scale=[0.3] * len(PARAMS), length_scale_bounds=(0.05, 5.0), nu=2.5)
              + WhiteKernel(0.05, (1e-4, 1.0)))
    proposals: list[tuple] = []
    seen = set(keys)
    for _ in range(n):
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2,
                                      random_state=int(rng.integers(1 << 31)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(X, y)
        candidates = rng.random((4000, len(PARAMS)))
        mu, sd = gp.predict(candidates, return_std=True)
        best = y.max()
        z = (mu - best) / np.maximum(sd, 1e-9)
        ei = (mu - best) * norm.cdf(z) + sd * norm.pdf(z)
        for idx in np.argsort(-ei):
            key = _from_unit(candidates[idx])
            if key not in seen:
                break
        seen.add(key)
        proposals.append(key)
        # Kriging believer: pretend the proposal scored its predicted mean, so the
        # next proposal in this batch goes somewhere else.
        X = np.vstack([X, _unit(key)])
        y = np.append(y, float(gp.predict(_unit(key)[None, :])[0]))
    return proposals


# ---------------------------------------------------------------- study

def _paired_bootstrap(a: pd.Series, b: pd.Series, n: int = 2000, block: int = 21, seed: int = 0) -> float:
    """P(Sharpe(a) - Sharpe(b) <= 0) under a paired circular block bootstrap."""
    rng = np.random.default_rng(seed)
    x, z = a.to_numpy(), b.to_numpy()
    t = len(x)
    starts = rng.integers(0, t, size=(n, int(np.ceil(t / block))))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :t] % t
    def sr(m):
        return m.mean(axis=1) / m.std(axis=1, ddof=1) * np.sqrt(252)
    return float((sr(x[idx]) - sr(z[idx]) <= 0).mean())


def run() -> dict[str, pd.DataFrame]:
    base = eras.FINAL
    ds = load_dataset(base)
    evaluate = Evaluator(ds, base)
    rng = np.random.default_rng(0)

    def grid_keys(grid):
        return [_key({"pead": p, "vsr_share": 0.5, "rebalance": r, "band": b})
                for p, r, b in itertools.product(grid["pead"], grid["rebalance"], grid["band"])]

    coarse, fine = grid_keys(COARSE), grid_keys(FINE)
    default = _key(DEFAULT)
    for key in fine + coarse + [default]:
        evaluate(key)
    log.warning("grids evaluated: %d configurations, %d portfolios", len(evaluate.returns), len(evaluate.portfolios))

    index = evaluate(default).index
    folds = make_folds(index, base.validation.train_days, base.validation.test_days,
                       base.validation.expanding, EMBARGO)

    # Optimizer state: start from the default plus a random design.
    bayes_points = [default] + [_from_unit(u) for u in rng.random((N_INIT, len(PARAMS)))]
    added = {k: 0 for k in bayes_points}

    methods = ["fixed default", "coarse argmax", "coarse neighbourhood", "fine argmax",
               "fine neighbourhood", "fine neighbourhood interior", "bayes argmax",
               "bayes neighbourhood"]
    oos = {m: [] for m in methods}
    rows = []

    for f, fold in enumerate(folds):
        lo, hi = fold.train_start, fold.train_end
        if f > 0:
            current = {k: _score(evaluate(k), lo, hi) for k in bayes_points}
            new = _propose(current, rng, N_PER_FOLD)
            bayes_points += new
            added.update({k: f for k in new})
        bayes_scores = {k: _score(evaluate(k), lo, hi) for k in bayes_points}
        coarse_scores = {k: _score(evaluate(k), lo, hi) for k in coarse}
        fine_scores = {k: _score(evaluate(k), lo, hi) for k in fine}

        chosen = {
            "fixed default": default,
            "coarse argmax": _best(coarse_scores),
            "coarse neighbourhood": _best(_grid_neighbourhood(coarse_scores, COARSE)),
            "fine argmax": _best(fine_scores),
            "fine neighbourhood": _best(_grid_neighbourhood(fine_scores, FINE)),
            "fine neighbourhood interior": _best(_grid_neighbourhood(fine_scores, FINE, interior=True)),
            "bayes argmax": _best(bayes_scores),
            "bayes neighbourhood": _best(_knn_neighbourhood(bayes_scores)),
        }
        for method, key in chosen.items():
            series = evaluate(key)
            test = series.loc[fold.test_start:fold.test_end]
            oos[method].append(test)
            rows.append({"fold": f, "test_start": fold.test_start.date(), "method": method,
                         **_params(key), "train_sharpe": _score(series, lo, hi),
                         "test_sharpe": sharpe(test)})
        log.warning("fold %2d: %d optimizer points", f, len(bayes_points))

    selections = pd.DataFrame(rows)
    stitched = {m: pd.concat(chunks) for m, chunks in oos.items()}
    summary = []
    for m in methods:
        sel = selections[selections.method == m]
        summary.append({
            "method": m,
            "oos_net_sharpe": sharpe(stitched[m]),
            "mean_train_sharpe": float(sel.train_sharpe.mean()),
            "mean_train_minus_test": float((sel.train_sharpe - sel.test_sharpe).mean()),
            "folds_positive": int((sel.test_sharpe > 0).sum()),
            "distinct_configs_chosen": int(sel[list(PARAMS)].drop_duplicates().shape[0]),
            "p_not_better_than_default": (np.nan if m == "fixed default" else
                                          _paired_bootstrap(stitched[m], stitched["fixed default"])),
        })
    trials = {"coarse": len(coarse), "fine": len(fine), "bayes": len(bayes_points)}
    summary = pd.DataFrame(summary)
    summary["configs_searched"] = summary.method.map(
        lambda m: 1 if m == "fixed default" else trials[m.split()[0]])

    points = pd.DataFrame([{**_params(k), "added_in_fold": added[k]} for k in bayes_points])

    # Full diagnostics for the headline rule. Every distinct configuration this
    # study evaluated is a trial, on top of the ledger's prior count.
    headline = stitched[HEADLINE]
    searched = list(dict.fromkeys(coarse + fine + bayes_points))
    trial_sharpes = pd.Series({str(k): _daily_sharpe(evaluate(k)) for k in searched})
    vc = base.validation
    dsr = v.deflated_sharpe(headline, PRIOR_TRIALS + len(searched), trial_sharpes)
    pbo = v.probability_of_backtest_overfitting(
        pd.DataFrame({str(k): evaluate(k) for k in fine}), vc.pbo_splits)
    boot = v.block_bootstrap_sharpe(headline, vc.bootstrap_samples, vc.bootstrap_block, seed=vc.seed)
    sel = selections[selections.method == HEADLINE]
    diagnostics = pd.concat([
        pd.Series({"oos_net_sharpe": sharpe(headline), "folds_positive": int((sel.test_sharpe > 0).sum()),
                   "mean_train_minus_test": float((sel.train_sharpe - sel.test_sharpe).mean()),
                   "prior_trials": PRIOR_TRIALS, "configs_this_study": len(searched)}),
        dsr.add_prefix("dsr_"), pbo.add_prefix("pbo_"), boot.add_prefix("boot_"),
    ]).rename("value").to_frame()
    diagnostics.index.name = "statistic"
    oos_returns = pd.DataFrame({m: stitched[m] for m in methods})
    oos_returns.index.name = "date"
    return {"bayes_summary": summary, "bayes_selections": selections, "bayes_points": points,
            "bayes_headline": diagnostics, "bayes_oos_returns": oos_returns}


def _daily_sharpe(series: pd.Series) -> float:
    sd = series.std(ddof=1)
    return float(series.mean() / sd) if sd else np.nan
