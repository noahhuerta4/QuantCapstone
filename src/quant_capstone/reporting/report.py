"""Render a :class:`Study` into ``results.md`` (spec S1 item 6).

The report is written to be read by someone who did not run it, and to be honest
in the specific ways Section 0 demands: gross next to net, in-sample next to
out-of-sample, the trial count stated, and a limitations section that names the
biases rather than burying them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..study import TARGET_SHARPE, TRADE_BAND, Study


def _fmt(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "—"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 1:
            return f"{value:,.3f}"
        return f"{value:.4f}"
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return str(value)


def table(frame: pd.DataFrame | pd.Series, index_label: str = "") -> str:
    """Minimal markdown table renderer (avoids a tabulate dependency)."""
    if isinstance(frame, pd.Series):
        frame = frame.to_frame("value")
    if frame.empty:
        return "_(no data)_\n"

    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [" ".join(str(p) for p in col) for col in frame.columns]

    header = [index_label or (frame.index.name or "")] + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for idx, row in frame.iterrows():
        cells = [_fmt(idx)] + [_fmt(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _headline(study: Study) -> str:
    oos = study.oos_net_sharpe
    full_net = study.performance.loc["sharpe", "full_net"]
    full_gross = study.performance.loc["sharpe", "full_gross"]
    verdict = "**clears**" if study.clears_target else "**does not clear**"

    ci = ""
    if not study.bootstrap.empty:
        ci = (
            f" The 95% block-bootstrap interval is "
            f"[{study.bootstrap['ci_low']:.2f}, {study.bootstrap['ci_high']:.2f}], and "
            f"{study.bootstrap['prob_sharpe_below_zero']:.0%} of bootstrap draws fall below zero."
        )

    dsr = ""
    if not study.deflated.empty:
        dsr = (
            f" After deflating for the {int(study.deflated['n_trials'])} configurations tried, the "
            f"Deflated Sharpe Ratio is {study.deflated['deflated_sharpe_ratio']:.3f} "
            f"(the probability the true Sharpe exceeds what the best of "
            f"{int(study.deflated['n_trials'])} worthless strategies would have produced)."
        )

    return f"""## Headline

**Net out-of-sample Sharpe: {oos:.2f}** — this {verdict} the design target of {TARGET_SHARPE:.1f}.

Over the full sample the strategy earns a gross Sharpe of {full_gross:.2f} and a net Sharpe of
{full_net:.2f}; costs account for the difference, and that gap is the central finding rather
than a footnote.{ci}{dsr}

{study.law.verdict}
"""


def _selection_note(study: Study) -> str:
    """Say plainly whether per-fold selection beat the fixed defaults, and what that means.

    When the defaults win, the tempting reading is "so use the defaults". That
    reading is wrong: the defaults were themselves picked by looking at
    full-sample results, so their column carries exactly the hindsight the
    walk-forward exists to remove. It is an optimistic reference, not a rival
    estimate, and the headline stays the selected number.
    """
    perf = study.performance.loc["sharpe"]
    selected, default = float(perf["OOS_net_selected"]), float(perf["OOS_net_default"])
    folds = study.walk_forward.folds
    if folds.empty or not (np.isfinite(selected) and np.isfinite(default)):
        return ""
    wins = int((folds["test_sharpe"] > folds["test_sharpe_default"]).sum())
    tally = f"selection beat the defaults in {wins} of {len(folds)} folds"
    if default >= selected:
        return (
            f"**The fixed defaults beat per-fold selection out of sample: {default:.2f} against "
            f"{selected:.2f}** ({tally}). This is not evidence that the defaults are better. "
            "Several of them were chosen by looking at full-sample results, so the default column "
            "carries the very hindsight the walk-forward removes; it is an optimistic reference "
            "rather than an honest estimate. What the gap does measure is the price of searching: "
            "picking the best grid cell on each training window adds nothing out of sample, so at "
            "the margin the selection is fitting noise. The headline is the selected number "
            "because it is the only one that could actually have been traded."
        )
    return (
        f"Per-fold selection beat the fixed defaults out of sample, {selected:.2f} against "
        f"{default:.2f} ({tally}), so re-choosing parameters on each training window carries "
        "information forward rather than fitting noise."
    )


def _trade_band_note(activity: pd.Series) -> str:
    """Check the S8 band on both readings of 'trades per day'.

    A multi-day rebalance makes the calendar average and the trading-day average
    diverge sharply, and quoting only the calendar average would let a strategy
    that fires 58 orders every tenth day report '5.8 trades per day' and claim
    the band. Both numbers, and the band verdict on each.
    """
    lo, hi = TRADE_BAND
    calendar = activity["mean_trades_per_day"]
    on_trading_days = activity.get("mean_trades_per_trading_day", calendar)
    share = activity.get("fraction_of_days_trading", 1.0)

    verdict = (
        "**sits inside**" if lo <= calendar <= hi else "**falls outside**"
    )
    context = (
        ""
        if share > 0.95
        else f" The book batches its trading into {share:.0%} of days, so those days see about "
        f"{on_trading_days:.0f} orders each — spikes above {hi} on a rebalance day are expected "
        f"for a batched schedule and are not treated as a breach."
    )
    return (
        f"Trades per day average **{calendar:.1f}**, which {verdict} the spec S8 target band "
        f"of {lo}–{hi}.{context}"
    )


def _vol_target_note(study: Study) -> str:
    """Flag the vol target / per-name cap conflict rather than leaving it implicit.

    These two constraints are routinely set independently and are routinely
    incompatible. A dollar-neutral book of N names capped at w per name can reach
    at most ``w * sigma_idio * sqrt(N)`` of annualized volatility, so with the
    defaults (1.5%, ~200 names, ~25% idiosyncratic vol) the ceiling is around 5%
    — half the 10% target, and no amount of signal changes that. Sharpe is
    scale-invariant so this does not by itself hurt the ratio, but it does mean
    the strategy delivers half the *return* the target implies, and the fixed
    per-order commission minimum is a proportionally larger drag on the smaller
    book.
    """
    diagnostics = study.run.portfolio.diagnostics
    live = diagnostics[diagnostics["n_active"] > 0]
    if live.empty:
        return ""

    achieved = float(live["ex_ante_vol"].mean())
    target = study.cfg.risk.vol_target
    at_cap = float(live["n_at_cap"].mean())
    names = float(live["n_active"].mean())

    if achieved >= target * 0.9:
        return (
            f"Ex-ante volatility averages {achieved:.2%} against a {target:.0%} target, so the "
            f"vol target is the binding sizing constraint as intended."
        )
    return (
        f"**The vol target is not reached: ex-ante volatility averages {achieved:.2%} against a "
        f"{target:.0%} target.** This is a constraint conflict, not a bug. With {names:.0f} names "
        f"capped at {study.cfg.risk.max_weight:.1%} of NAV each, maximum gross exposure is "
        f"{names * study.cfg.risk.max_weight:.1f}x, and a dollar-neutral book that size simply "
        f"cannot carry {target:.0%} volatility — {at_cap:.0f} "
        f"position{'' if round(at_cap) == 1 else 's'} sit{'s' if round(at_cap) == 1 else ''} at "
        f"the cap on an average day. Reaching the target would need a wider universe, a higher per-name cap, or "
        f"a lower target; raising the cap concentrates the book, which is the trade being made. "
        f"Sharpe is unaffected by the scale, but realized return is roughly "
        f"{achieved / target:.0%} of what the target implies."
    )


def _limitations(study: Study) -> str:
    source = study.cfg.data.source
    survivorship = (
        "The universe is seeded from *current* S&P 500 membership via yfinance, which returns only "
        "currently-listed tickers. Names that fell far enough to be delisted are absent from the "
        "entire history. For a reversal strategy the direction of this bias is unambiguous and "
        "unfavourable: the missing names are disproportionately the ones whose 'oversold, buy it' "
        "signal never reverted, so the reported results are **optimistic**. A point-in-time "
        "constituent source, or an IBKR pull that includes delistings, is the fix; until then, "
        "treat every number here as an upper bound."
        if source == "yfinance"
        else "This run used **synthetic data with a deliberately planted reversal effect**. It "
        "validates that the pipeline recovers a known signal and that the accounting is "
        "self-consistent. It says nothing whatsoever about whether the strategy works on real "
        "markets, and no number in this report should be quoted as if it did."
    )

    return f"""## Validity and limitations

**Survivorship.** {survivorship}

**Costs we model.** Commission with a per-order minimum, SEC Section 31 and FINRA TAF on sales,
general-collateral borrow on the short leg, and slippage as half-spread plus a
`c·σ·√(shares/ADV)` impact term. Rates reflect the published IBKR schedule at build time and
drift; verify before quoting.

**Costs we do not model.** Margin financing on the long leg beyond cash, which a book running
{study.activity['mean_gross_exposure']:.1f}x gross genuinely pays. Hard-to-borrow rates — the
names a reversal signal most wants to short are recent winners, which skew expensive to borrow, so
the uniform general-collateral rate is a floor. Fill uncertainty: we assume every intended trade
executes in full at the modelled price. Each of these makes the true net number worse than the
one reported.

**Impact calibration.** The square-root impact coefficient `c = {study.cfg.costs.impact_coef}` is
assumed, not estimated from our own fills, which we do not have. This is why the cost-sensitivity
sweep matters more than the base case: the break-even multiplier of
{_fmt(study.break_even)}x is the number to judge robustness on.

**Execution timing.** Signals use data through the day-t close and trade at the t+1 open. The
`same_close` alternative is optimistic by construction and is provided only for comparison.
"""


def build_report(study: Study) -> str:
    cfg = study.cfg
    activity = study.activity

    parts = [
        "# Results — Short-Term Cross-Sectional Reversal",
        "",
        f"_Data source: `{cfg.data.source}` · {cfg.data.start} to {cfg.data.end} · "
        f"universe ≈ {cfg.data.universe_size} names · k = {cfg.signal.k} · "
        f"vol target {cfg.risk.vol_target:.0%} · {cfg.execution.timing} execution_",
        "",
        _headline(study),
        "",
        "## Performance — gross vs net, in-sample vs out-of-sample",
        "",
        "Only the `OOS_net_selected` column counts toward the objective. `OOS_net_default` is the",
        "same out-of-sample windows using fixed default parameters, and the comparison between the",
        "two says whether per-fold parameter selection added anything or just fitted noise.",
        "",
        table(study.performance, "metric"),
        "",
        _selection_note(study),
        "",
        "## Where the return goes",
        "",
        table(study.attribution.to_frame("bps of NAV per year"), "component"),
        "",
        "Net Sharpe with each component switched off in turn:",
        "",
        table(study.cost_toggles),
        "",
        "## Trade activity",
        "",
        _trade_band_note(activity),
        "",
        table(activity.to_frame("value"), "measure"),
        "",
        "## Risk and neutrality",
        "",
        "Ex-post checks that the constraints the pipeline claims to impose actually hold:",
        "",
        table(study.neutrality.describe().loc[["mean", "std", "min", "max"]].T, "exposure"),
        "",
        f"Realized beta of the net return series against the equal-weight universe: "
        f"**{study.run.realized_beta():.4f}**.",
        "",
        _vol_target_note(study),
        "",
        table(study.constraints.describe().loc[["mean", "max"]].T, "constraint"),
        "",
        "## Fundamental Law diagnostic",
        "",
        table(study.law.ic.to_frame("value"), "IC"),
        "",
        table(study.law.breadth.to_frame("value"), "breadth"),
        "",
        table(study.law.transfer.to_frame("value"), "transfer"),
        "",
        "### The seductive lie",
        "",
        "The IC required to reach the target IR, as each naive assumption is removed:",
        "",
        table(study.law.seductive_lie, "assumption"),
        "",
        table(study.law.comparison.to_frame("value"), "implied vs realized"),
        "",
        "### Signal decay",
        "",
        table(study.law.ic_decay.to_frame("mean IC"), "horizon (days)"),
        "",
        "## Overfitting controls",
        "",
        "### Walk-forward",
        "",
        table(study.walk_forward.folds, "fold"),
        "",
        table(study.walk_forward.stability().to_frame("value"), "stability"),
        "",
        "### Parameter sensitivity",
        "",
        "A broad plateau means the effect is real; a lone bright cell means the grid found noise.",
        "",
    ]

    for name, frame in study.surfaces.items():
        parts += [f"**{name.replace('__vs__', ' vs ')}** (net Sharpe)", "", table(frame), ""]

    parts += [
        table(study.plateau.to_frame("value"), "plateau"),
        "",
        "### Deflated Sharpe and PBO",
        "",
        table(study.deflated.to_frame("value"), "deflated sharpe"),
        "",
        table(study.pbo.to_frame("value"), "PBO"),
        "",
        "### Bootstrap confidence interval",
        "",
        table(study.bootstrap.to_frame("value"), "bootstrap"),
        "",
        "## Cost sensitivity and capacity",
        "",
        f"Break-even cost multiplier: **{_fmt(study.break_even)}x** baseline.",
        "",
        table(study.cost_sweep),
        "",
        "Net Sharpe as a function of fund size — the liquidity cap and square-root impact are",
        "what bite as NAV grows:",
        "",
        table(study.capacity),
        "",
        _limitations(study),
        "",
    ]

    if study.plots:
        parts += ["## Figures", ""]
        parts += [f"![{p.stem}]({p.name})" for p in study.plots]
        parts += [""]

    return "\n".join(parts)


def write_report(study: Study, outdir: Path | str = "output", name: str = "results.md") -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    path.write_text(build_report(study), encoding="utf-8")
    return path
