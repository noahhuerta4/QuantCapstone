"""Report figures (spec S11).

Each function takes what it needs, saves a PNG, and returns the path. Form is
chosen by what the data has to say, not by habit: equity and exposure are
change-over-time (lines), cost attribution is magnitude across a handful of
labelled categories (one-hue bars, since the labels carry identity), the
parameter sweep is a signed surface (diverging, gray at zero).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import TRADING_DAYS
from .metrics import drawdown_series, equity_curve, rolling_sharpe
from .style import (
    DIVERGING,
    GRID,
    NEGATIVE,
    NEUTRAL_BAND,
    SEQUENTIAL,
    SERIES,
    TEXT_MUTED,
    TEXT_SECONDARY,
    finish,
    use_style,
)


def _save(fig, outdir: Path, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def equity_and_drawdown(gross: pd.Series, net: pd.Series, outdir: Path) -> Path:
    """The headline chart: gross and net on one scale, drawdown beneath.

    Both curves share an axis because they are the same measure -- growth of the
    same dollar -- and the vertical gap between them *is* the cost of trading,
    which is the finding the whole project is about.
    """
    use_style()
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.6), sharex=True, height_ratios=[2.6, 1], constrained_layout=True
    )

    for series, color, name in ((gross, SERIES[0], "Gross"), (net, SERIES[1], "Net of costs")):
        curve = equity_curve(series)
        ax.plot(curve.index, curve.to_numpy(), color=color, label=name)
        if len(curve):
            ax.annotate(
                f"{name}  {curve.iloc[-1]:.2f}x",
                xy=(curve.index[-1], curve.iloc[-1]),
                xytext=(6, 0), textcoords="offset points",
                color=color, fontsize=8.5, va="center",
            )
    ax.axhline(1.0, color=GRID, linewidth=0.8, zorder=0)
    ax.legend(loc="upper left")
    finish(ax, "Growth of $1", "The gap between the lines is transaction cost", "multiple")

    dd = drawdown_series(net) * 100
    ax2.fill_between(dd.index, dd.to_numpy(), 0, color=NEGATIVE, alpha=0.22, linewidth=0)
    ax2.plot(dd.index, dd.to_numpy(), color=NEGATIVE, linewidth=1.2)
    finish(ax2, "Net drawdown", ylabel="%")
    return _save(fig, outdir, "equity_and_drawdown")


def portfolio_evolution(
    curves: dict[str, pd.Series], outdir: Path, start_value: float = 1_000_000.0
) -> Path:
    """Account value over time, with drawdown beneath.

    Dollars rather than a growth multiple, because the point of the chart is what
    the account is actually worth. The drawdown panel is not decoration: it is the
    experience of holding the thing, and a curve that ends higher after two years
    underwater is a different proposition from one that does not.

    A single series carries no legend -- the title names it.
    """
    use_style()
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.8), sharex=True, height_ratios=[2.4, 1], constrained_layout=True
    )

    for (label, curve), color in zip(curves.items(), SERIES):
        ax.plot(curve.index, curve.to_numpy(), color=color, label=label)
        ax.annotate(
            f"{label}   ${curve.iloc[-1]:,.0f}",
            xy=(curve.index[-1], curve.iloc[-1]), xytext=(6, 0),
            textcoords="offset points", color=color, fontsize=8.5, va="center",
        )
    ax.axhline(start_value, color=GRID, linewidth=0.9, zorder=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    if len(curves) > 1:
        ax.legend(loc="upper left", ncols=len(curves))
    ax.margins(x=0.10)
    finish(ax, "Portfolio value", f"Starting from ${start_value:,.0f}", "account value")

    for (label, curve), color in zip(curves.items(), SERIES):
        dd = (curve / curve.cummax() - 1.0) * 100
        if len(curves) == 1:
            ax2.fill_between(dd.index, dd.to_numpy(), 0, color=color, alpha=0.18, linewidth=0)
        ax2.plot(dd.index, dd.to_numpy(), color=color, linewidth=1.2, label=label)
    ax2.axhline(0, color=GRID, linewidth=0.9)
    worst = min(float((c / c.cummax() - 1).min()) for c in curves.values())
    finish(ax2, "Drawdown", f"Worst peak-to-trough decline: {worst:.1%}", "%")
    return _save(fig, outdir, "portfolio_evolution")


def rolling_sharpe_plot(net: pd.Series, outdir: Path, window: int = 126) -> Path:
    """Rolling 6-month Sharpe -- where regime fragility becomes visible."""
    use_style()
    fig, ax = plt.subplots(figsize=(9, 3.4), constrained_layout=True)
    rs = rolling_sharpe(net, window).dropna()

    ax.fill_between(rs.index, rs.to_numpy(), 0, where=rs.to_numpy() < 0,
                    color=NEGATIVE, alpha=0.15, linewidth=0)
    ax.plot(rs.index, rs.to_numpy(), color=SERIES[0])
    ax.axhline(0, color=TEXT_MUTED, linewidth=0.9)
    finish(
        ax,
        f"Rolling {window}-day net Sharpe",
        f"Positive in {(rs > 0).mean():.0%} of rolling windows; mean {rs.mean():.2f}",
        "Sharpe",
    )
    return _save(fig, outdir, "rolling_sharpe")


def cost_attribution(attribution: pd.Series, outdir: Path) -> Path:
    """Annualized drag by component. One hue: the axis labels carry identity."""
    use_style()
    components = attribution.drop(labels=["total"], errors="ignore").sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 3.0), constrained_layout=True)

    labels = [c.replace("_", " ") for c in components.index]
    ax.barh(labels, components.to_numpy(), color=SEQUENTIAL, height=0.62)
    for y, value in enumerate(components.to_numpy()):
        ax.annotate(
            f"{value:,.0f}", xy=(value, y), xytext=(5, 0), textcoords="offset points",
            va="center", fontsize=8.5, color=TEXT_SECONDARY,
        )
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0, max(components.max() * 1.18, 1))
    total = float(attribution.get("total", components.sum()))
    finish(ax, "Where the return goes", f"Total {total:,.0f} bps of NAV per year")
    # Horizontal bars: the magnitude is on x, so the units label belongs there.
    ax.set_xlabel("bps / year")
    return _save(fig, outdir, "cost_attribution")


def exposures(exposure_frame: pd.DataFrame, outdir: Path) -> Path:
    """Long, short and net over time. Net sitting on zero is the neutrality proof."""
    use_style()
    fig, ax = plt.subplots(figsize=(9, 3.2), constrained_layout=True)
    for column, color in zip(("long", "short", "net"), SERIES):
        if column in exposure_frame:
            ax.plot(
                exposure_frame.index, exposure_frame[column].to_numpy() * 100,
                color=color, label=column, linewidth=1.4,
            )
    ax.axhline(0, color=GRID, linewidth=0.8)
    ax.legend(loc="center left", ncols=3)
    finish(ax, "Exposure", "Net pinned to zero by construction", "% of NAV")
    return _save(fig, outdir, "exposures")


def trades_per_day(n_trades: pd.Series, outdir: Path, band: tuple[int, int] = (5, 50)) -> Path:
    """Trade count against the target band (spec S8)."""
    use_style()
    fig, ax = plt.subplots(figsize=(9, 3.0), constrained_layout=True)
    smooth = n_trades.rolling(21, min_periods=5).mean()

    ax.axhspan(band[0], band[1], color=NEUTRAL_BAND, zorder=0)
    ax.plot(n_trades.index, n_trades.to_numpy(), color=GRID, linewidth=0.7, zorder=1)
    ax.plot(smooth.index, smooth.to_numpy(), color=SERIES[0], zorder=2)
    inside = n_trades[(n_trades >= band[0]) & (n_trades <= band[1])]
    finish(
        ax,
        "Trades per day",
        f"{len(inside) / max(len(n_trades), 1):.0%} of days inside the "
        f"{band[0]}-{band[1]} band (21-day average in blue)",
        "trades",
    )
    return _save(fig, outdir, "trades_per_day")


def sensitivity_heatmap(
    surface_frame: pd.DataFrame, outdir: Path, metric: str = "net Sharpe", name: str = "sensitivity"
) -> Path:
    """Diverging surface centred on zero: a plateau is good, a lone bright cell is not."""
    use_style()
    values = surface_frame.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(values))) or 1.0

    fig, ax = plt.subplots(figsize=(1.15 * values.shape[1] + 3.0, 0.62 * values.shape[0] + 2.2),
                           constrained_layout=True)
    mesh = ax.imshow(values, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")

    ax.set_xticks(range(values.shape[1]), [str(c) for c in surface_frame.columns])
    ax.set_yticks(range(values.shape[0]), [str(i) for i in surface_frame.index])
    ax.set_xlabel(str(surface_frame.columns.name or ""))
    ax.set_ylabel(str(surface_frame.index.name or ""))
    ax.grid(visible=False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                shade = "#ffffff" if abs(values[i, j]) > limit * 0.55 else "#0b0b0b"
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                        fontsize=8.5, color=shade)

    fig.colorbar(mesh, ax=ax, shrink=0.82, label=metric)
    finish(ax, f"Parameter sensitivity — {metric}", "Look for a plateau, not a peak")
    return _save(fig, outdir, name)


def cost_sensitivity_plot(sweep: pd.DataFrame, outdir: Path, break_even: float | None = None) -> Path:
    """Net Sharpe as costs are scaled up; the zero crossing is the robustness margin."""
    use_style()
    fig, ax = plt.subplots(figsize=(6.6, 3.4), constrained_layout=True)
    ax.plot(sweep.index, sweep["net_sharpe"].to_numpy(), color=SERIES[0], marker="o")
    ax.axhline(0, color=GRID, linewidth=0.9)
    ax.axvline(1.0, color=TEXT_MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.annotate("baseline", xy=(1.0, ax.get_ylim()[1]), xytext=(4, -10),
                textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)

    subtitle = "Net Sharpe against a global multiplier on every cost"
    if break_even is not None and np.isfinite(break_even):
        ax.axvline(break_even, color=NEGATIVE, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.annotate(f"break-even {break_even:.1f}x", xy=(break_even, 0), xytext=(5, 8),
                    textcoords="offset points", fontsize=8.5, color=NEGATIVE)
        subtitle = f"Edge survives to {break_even:.1f}x assumed costs"
    finish(ax, "Cost sensitivity", subtitle, "net Sharpe")
    ax.set_xlabel("cost multiplier")
    return _save(fig, outdir, "cost_sensitivity")


def bootstrap_plot(draws: np.ndarray, point: float, ci: tuple[float, float], outdir: Path) -> Path:
    """Sampling distribution of the Sharpe. The width is the point."""
    use_style()
    fig, ax = plt.subplots(figsize=(6.6, 3.2), constrained_layout=True)
    ax.hist(draws, bins=48, color=SEQUENTIAL, alpha=0.85, edgecolor="none")
    ax.axvline(point, color=TEXT_SECONDARY, linewidth=1.4)
    for bound in ci:
        ax.axvline(bound, color=NEGATIVE, linewidth=1.1, linestyle=(0, (4, 3)))
    ax.grid(axis="x", visible=False)
    finish(
        ax,
        "Block-bootstrap Sharpe distribution",
        f"Point estimate {point:.2f}, 95% interval [{ci[0]:.2f}, {ci[1]:.2f}]",
        "draws",
    )
    ax.set_xlabel("annualized Sharpe")
    return _save(fig, outdir, "bootstrap_sharpe")


def ic_plot(ic_series: pd.Series, outdir: Path) -> Path:
    """Cumulative IC -- a straight rising line means a stable forecast."""
    use_style()
    clean = ic_series.dropna()
    fig, ax = plt.subplots(figsize=(9, 3.0), constrained_layout=True)
    ax.plot(clean.index, clean.cumsum().to_numpy(), color=SERIES[0])
    ax.axhline(0, color=GRID, linewidth=0.8)
    finish(
        ax,
        "Cumulative information coefficient",
        f"Mean IC {clean.mean():.4f}, positive on {(clean > 0).mean():.0%} of days",
        "cumulative IC",
    )
    return _save(fig, outdir, "information_coefficient")


def walk_forward_plot(folds: pd.DataFrame, outdir: Path) -> Path:
    """Per-fold train vs test Sharpe. The gap is the overfitting premium."""
    use_style()
    fig, ax = plt.subplots(figsize=(9, 3.4), constrained_layout=True)
    x = np.arange(len(folds))
    width = 0.38

    ax.bar(x - width / 2, folds["train_sharpe"], width * 0.94, color=SERIES[0], label="train (IS)")
    ax.bar(x + width / 2, folds["test_sharpe"], width * 0.94, color=SERIES[1], label="test (OOS)")
    ax.axhline(0, color=TEXT_MUTED, linewidth=0.9)
    ax.set_xticks(x, [str(f) for f in folds["fold"]], rotation=30, ha="right", fontsize=7.5)
    ax.legend(loc="best", ncols=2)
    gap = float((folds["train_sharpe"] - folds["test_sharpe"]).mean())
    finish(ax, "Walk-forward folds",
           f"Only the orange bars are out-of-sample; mean train-minus-test gap {gap:+.2f}",
           "Sharpe")
    return _save(fig, outdir, "walk_forward")


def all_plots(run, outdir: Path, extras: dict | None = None) -> list[Path]:
    """Every standard figure for one strategy run."""
    from ..costs.model import attribution_bps

    outdir = Path(outdir)
    result = run.result
    paths = [
        equity_and_drawdown(result.gross_return, result.net_return, outdir),
        rolling_sharpe_plot(result.net_return, outdir),
        cost_attribution(attribution_bps(result.costs, result.nav), outdir),
        exposures(result.exposures, outdir),
        trades_per_day(result.n_trades, outdir),
    ]
    for key, fn in (("ic_series", ic_plot),):
        if extras and key in extras:
            paths.append(fn(extras[key], outdir))
    return paths


# ----------------------------------------------------------- analysis figures
def sleeve_correlation_heatmap(corr: pd.DataFrame, outdir: Path, name: str = "sleeve_correlations") -> Path:
    """Diverging matrix of cross-sleeve correlation.

    Diverging rather than sequential because the sign is meaningful: a negative
    correlation between sleeves is more valuable than a zero one, and a
    sequential ramp would render those two the same shade.
    """
    use_style()
    values = corr.to_numpy(dtype=float)
    labels = [c.replace("_", " ") for c in corr.columns]

    fig, ax = plt.subplots(figsize=(1.05 * len(labels) + 3.2, 0.85 * len(labels) + 2.2),
                           constrained_layout=True)
    mesh = ax.imshow(values, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.grid(visible=False)

    for i in range(len(labels)):
        for j in range(len(labels)):
            v = values[i, j]
            shade = "#ffffff" if abs(v) > 0.55 else "#0b0b0b"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=shade)

    fig.colorbar(mesh, ax=ax, shrink=0.8, label="correlation")
    finish(ax, "Sleeve correlations",
           "Blocks on the diagonal are redundant variants; near-zero off-block is real diversification")
    return _save(fig, outdir, name)


def sleeve_scatter(sleeves: pd.DataFrame, outdir: Path) -> Path:
    """IC against how fast a sleeve turns over -- the two terms that set its IR.

    The x-axis is ``1 - autocorrelation`` on a log scale rather than the raw
    autocorrelation. Two reasons: the quantity is proportional to the number of
    independent periods a signal delivers, so it is the term that actually enters
    ``IR = IC * sqrt(BR)``; and the sleeves that matter most are bunched between
    0.99 and 0.9995 autocorrelation, which a linear axis renders as one
    indistinguishable column.

    Useful sleeves sit upper-right: forecasting power *and* a signal that keeps
    moving. Anything on the left edge has no breadth however good its IC.
    """
    use_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)

    turnover = (1.0 - sleeves["autocorr"]).clip(lower=1e-4)
    used = sleeves["used"].to_numpy(dtype=bool) if "used" in sleeves else np.ones(len(sleeves), bool)

    for mask, color, label in ((used, SERIES[0], "in the blend"), (~used, TEXT_MUTED, "rejected")):
        if mask.any():
            ax.scatter(turnover[mask], sleeves["ic_mean"][mask], s=115, color=color,
                       label=label, zorder=3, edgecolor="#fcfcfb", linewidth=1.2)

    # Alternate label sides so the cluster of slow sleeves stays readable.
    order = np.argsort(turnover.to_numpy())
    for rank, i in enumerate(order):
        row = sleeves.iloc[i]
        above = rank % 2 == 0
        ax.annotate(str(row["sleeve"]).replace("_", " "),
                    xy=(turnover.iloc[i], row["ic_mean"]),
                    xytext=(0, 13 if above else -17), textcoords="offset points",
                    ha="center", fontsize=7.5,
                    color=SERIES[0] if used[i] else TEXT_SECONDARY)

    ax.set_xscale("log")
    ax.axhline(0, color=TEXT_MUTED, linewidth=0.9)
    ax.set_xlabel("1 − autocorrelation, log scale  (left = barely changes, no breadth)")
    ax.legend(loc="upper left", ncols=2)
    ax.margins(y=0.22)
    finish(ax, "Why some sleeves make the blend and others do not",
           "IR = IC × √breadth, so a strong IC on the left edge converts to almost nothing",
           "information coefficient")
    return _save(fig, outdir, "sleeve_scatter")


def breadth_waterfall(breadth: pd.Series, outdir: Path) -> Path:
    """Naive bet count collapsing to the effective one, on a log scale.

    Log because the corrections span an order of magnitude, and a linear axis
    would render every bar after the first as an indistinguishable stub.
    """
    use_style()
    naive = float(breadth["breadth_naive"])
    corr_adjusted = float(breadth["n_eff_after_factor_removal"]) * TRADING_DAYS
    effective = float(breadth["breadth_effective"])

    labels = ["naive\n(names x days)", "after cross-sectional\ndependence",
              "after signal\npersistence"]
    values = [naive, corr_adjusted, effective]

    fig, ax = plt.subplots(figsize=(6.8, 3.6), constrained_layout=True)
    ax.bar(labels, values, color=[SEQUENTIAL, SEQUENTIAL, SERIES[1]], width=0.58)
    ax.set_yscale("log")
    # Headroom so the value label on the tallest bar clears the subtitle.
    ax.set_ylim(min(values) * 0.45, max(values) * 2.6)
    for x, v in enumerate(values):
        ax.annotate(f"{v:,.0f}", xy=(x, v), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9, color=TEXT_SECONDARY)
    ax.grid(axis="x", visible=False)
    finish(ax, "Effective breadth",
           f"{naive / effective:,.0f}x fewer independent bets than the naive count suggests",
           "independent bets per year (log scale)")
    return _save(fig, outdir, "breadth_waterfall")


def liquidity_deciles(summary: pd.DataFrame, outdir: Path) -> Path:
    """Alpha and cost across liquidity, stacked rather than on twin axes.

    Two panels sharing an x-axis, never two y-scales on one plot: IC and spread
    have unrelated units, and overlaying them would invite a comparison of slopes
    that means nothing.
    """
    use_style()
    # Exclude the "Q1 minus Q10" summary row; it is a difference, not a decile.
    rows = summary[summary.index.str.match(r"^Q\d+$")]
    x = np.arange(len(rows))
    labels = [i.replace("Q", "") for i in rows.index]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.6, 4.8), sharex=True,
                                  height_ratios=[1, 1], constrained_layout=True)
    ax.bar(x, rows["ic_mean"], color=SERIES[0], width=0.6)
    ax.axhline(float(rows["ic_mean"].mean()), color=TEXT_MUTED,
               linewidth=0.9, linestyle=(0, (4, 3)))
    finish(ax, "Alpha does not improve as liquidity falls",
           "Reversal IC by dollar-volume decile (1 = least liquid); dashed line is the mean",
           "IC")

    ax2.bar(x, rows["half_spread_bps"], color=SERIES[1], width=0.6)
    ax2.set_xticks(x, labels)
    ax2.set_xlabel("dollar-volume decile  (1 = least liquid, 10 = most)")
    finish(ax2, "But cost does",
           "Bar-based spread estimate: the ordering is reliable, the absolute levels are not",
           "half-spread (bps)")
    return _save(fig, outdir, "liquidity_deciles")


def weight_sweep(sweep: pd.DataFrame, outdir: Path, x: str, chosen: float | None = None) -> Path:
    """Net Sharpe against one blend weight -- plateau versus spike."""
    use_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.4), constrained_layout=True)
    ax.plot(sweep[x], sweep["net"], color=SERIES[0], marker="o")
    ax.axhline(float(sweep["net"].iloc[0]), color=TEXT_MUTED, linewidth=0.9,
               linestyle=(0, (4, 3)))
    ax.annotate("without it", xy=(sweep[x].iloc[-1], sweep["net"].iloc[0]),
                xytext=(-4, 6), textcoords="offset points", ha="right",
                fontsize=8, color=TEXT_SECONDARY)
    if chosen is not None:
        ax.axvline(chosen, color=NEGATIVE, linewidth=1.1, linestyle=(0, (4, 3)))
        ax.annotate(f"chosen {chosen:.0%}", xy=(chosen, sweep["net"].max()),
                    xytext=(5, -2), textcoords="offset points", fontsize=8.5, color=NEGATIVE)
    ax.set_xlabel("weight on the earnings-surprise sleeve")
    finish(ax, "Blend weight sensitivity",
           "A flat region either side of the pick, not a single lucky cell", "net Sharpe")
    return _save(fig, outdir, "pead_weight")


def ic_decay_plot(decay: pd.Series, outdir: Path) -> Path:
    """How fast the forecast loses its power -- this sets the holding period."""
    use_style()
    fig, ax = plt.subplots(figsize=(6.6, 3.2), constrained_layout=True)
    ax.plot(decay.index, decay.to_numpy(), color=SERIES[0], marker="o")
    ax.axhline(0, color=TEXT_MUTED, linewidth=0.9)
    ax.set_xlabel("days ahead")
    finish(ax, "Signal decay", "Mean IC against returns h days forward", "IC")
    return _save(fig, outdir, "ic_decay")
