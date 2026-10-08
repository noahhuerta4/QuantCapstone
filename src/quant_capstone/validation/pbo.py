"""Probability of Backtest Overfitting via Combinatorially Symmetric
Cross-Validation (Bailey, Borwein, Lopez de Prado & Zhu; spec S10).

The procedure: chop the sample into S blocks, take every way of splitting them
into equal in-sample and out-of-sample halves, and for each split pick the
configuration that won in-sample and look up where it ranks out-of-sample. If
selection carries real information, the IS winner should land in the upper half
OOS. PBO is the fraction of splits where it lands in the *lower* half.

Reading it: PBO near 0.5 means selecting on in-sample performance is a coin flip
— whatever the backtest found is noise. Below ~0.2 suggests the ranking carries
signal. Note that this measures the *selection process*, not any single
configuration, so it is the right test for 'did I overfit by sweeping', which is
exactly the risk this project runs.

Implementation note: Sharpes on an arbitrary union of blocks are reconstructed
from per-block sums and sums of squares, so the C(14,7) = 3432 splits cost almost
nothing instead of 3432 full passes over the data.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def _block_moments(matrix: np.ndarray, n_blocks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-block count, sum and sum-of-squares for every configuration."""
    n_obs = matrix.shape[0]
    edges = np.linspace(0, n_obs, n_blocks + 1, dtype=int)
    counts = np.empty(n_blocks)
    sums = np.empty((n_blocks, matrix.shape[1]))
    sumsq = np.empty((n_blocks, matrix.shape[1]))
    for b in range(n_blocks):
        chunk = matrix[edges[b] : edges[b + 1]]
        counts[b] = len(chunk)
        sums[b] = chunk.sum(axis=0)
        sumsq[b] = (chunk**2).sum(axis=0)
    return counts, sums, sumsq


def _sharpe_from_moments(n: float, total: np.ndarray, total_sq: np.ndarray) -> np.ndarray:
    if n < 3:
        return np.full(total.shape, np.nan)
    mean = total / n
    variance = (total_sq - n * mean**2) / (n - 1)
    std = np.sqrt(np.clip(variance, 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(std > 0, mean / std, np.nan)


def probability_of_backtest_overfitting(
    returns: pd.DataFrame, n_blocks: int = 14
) -> pd.Series:
    """PBO plus the supporting degradation statistics.

    One output needs a health warning: ``oos_on_is_slope`` tends toward -1 by
    construction, because IS and OOS are complementary halves of one fixed
    sample, so a split that happens to put the good days in-sample necessarily
    leaves the bad ones out. Read it as a diagnostic of split noise, not as
    evidence of overfitting; ``pbo`` itself is the number that carries meaning.
    """
    clean = returns.dropna(how="all").fillna(0.0)
    n_configs = clean.shape[1]
    if n_configs < 2:
        return pd.Series({"pbo": np.nan, "n_configs": float(n_configs)})
    if n_blocks % 2:
        n_blocks -= 1
    n_blocks = max(min(n_blocks, len(clean) // 10), 4)

    matrix = clean.to_numpy(dtype=float)
    counts, sums, sumsq = _block_moments(matrix, n_blocks)
    all_blocks = set(range(n_blocks))

    logits, is_sharpes, oos_sharpes = [], [], []
    for combo in itertools.combinations(range(n_blocks), n_blocks // 2):
        is_idx = list(combo)
        oos_idx = list(all_blocks - set(combo))

        is_sr = _sharpe_from_moments(counts[is_idx].sum(), sums[is_idx].sum(0), sumsq[is_idx].sum(0))
        oos_sr = _sharpe_from_moments(counts[oos_idx].sum(), sums[oos_idx].sum(0), sumsq[oos_idx].sum(0))
        if not np.isfinite(is_sr).any() or not np.isfinite(oos_sr).any():
            continue

        best = int(np.nanargmax(is_sr))
        # Relative rank of the IS winner within the OOS Sharpes.
        order = pd.Series(oos_sr).rank(method="average", na_option="bottom").to_numpy()
        omega = order[best] / (n_configs + 1)
        omega = float(np.clip(omega, 1e-6, 1 - 1e-6))

        logits.append(np.log(omega / (1 - omega)))
        is_sharpes.append(is_sr[best])
        oos_sharpes.append(oos_sr[best])

    if not logits:
        return pd.Series({"pbo": np.nan, "n_configs": float(n_configs)})

    logits = np.asarray(logits)
    is_arr, oos_arr = np.asarray(is_sharpes), np.asarray(oos_sharpes)
    finite = np.isfinite(is_arr) & np.isfinite(oos_arr)
    slope = (
        float(np.polyfit(is_arr[finite], oos_arr[finite], 1)[0]) if finite.sum() > 2 else np.nan
    )

    return pd.Series(
        {
            "pbo": float((logits <= 0).mean()),
            "mean_logit": float(logits.mean()),
            "n_splits": float(len(logits)),
            "n_blocks": float(n_blocks),
            "n_configs": float(n_configs),
            "mean_is_sharpe_of_winner": float(np.nanmean(is_arr)),
            "mean_oos_sharpe_of_winner": float(np.nanmean(oos_arr)),
            "sharpe_degradation": float(np.nanmean(is_arr) - np.nanmean(oos_arr)),
            "oos_on_is_slope": slope,
            "prob_oos_loss": float(np.mean(oos_arr[finite] < 0)) if finite.any() else np.nan,
        }
    )
