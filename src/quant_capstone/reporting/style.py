"""Chart palette and matplotlib defaults.

Colors are the validated reference palette. Charts here never put more than
three categorical series on one axes, which is the subset documented as clearing
the all-pairs colorblind gates (CVD dE 9.2, normal-vision 24.0 on the light
surface) rather than only the adjacent-pair gates.

Rules this module enforces so individual plots do not have to think about them:

* **Color by the job.** Identity -> ``SERIES``. Magnitude of one measure ->
  ``SEQUENTIAL`` (a single hue; the category labels on the axis carry identity,
  so a rainbow of bars would be color doing no work). Signed quantities with a
  meaningful zero -> ``DIVERGING``, gray at the midpoint.
* **Never two y-axes.** If two measures do not share a scale they get two axes
  objects, not one with a twin.
* **Recessive chrome.** Thin horizontal-only grid behind the data, no top or
  right spine, muted tick labels.

These render to PNG for the report, so the palette commits to the light surface
only. Every chart ships with the equivalent table in ``results.md``, which is
also what satisfies the relief rule for the lower-contrast slots.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8985"
GRID = "#e6e5e1"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
"""Categorical slots 1-3, in fixed order. A fourth series folds into 'Other'."""

SEQUENTIAL = "#2a78d6"
NEGATIVE = "#e34948"
NEUTRAL_BAND = "#f0efec"

DIVERGING = LinearSegmentedColormap.from_list(
    "quant_diverging",
    ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#f0a09f", "#e34948", "#8f2020"],
)

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",  # DejaVu Sans ships no 600 weight
    "axes.titlelocation": "left",
    "axes.titlepad": 10,
    "axes.labelsize": 9,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "grid.alpha": 1.0,
    "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "lines.linewidth": 1.8,
    "lines.markersize": 4.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.handlelength": 1.6,
}


def use_style() -> None:
    plt.rcParams.update(RC)


SUBTITLE_PAD = 26
"""Title offset in points when a subtitle sits beneath it.

The default 10pt pad puts the title baseline right where the subtitle's ascender
lands, so the two overlap. The subtitle is drawn in axes coordinates and the
title in points, so the clearance has to be set in points on the title side.
"""


def finish(ax, title: str, subtitle: str = "", ylabel: str = "") -> None:
    """Title, optional subtitle and recessive chrome, applied consistently."""
    ax.set_title(title, color=TEXT_PRIMARY, pad=SUBTITLE_PAD if subtitle else None)
    if subtitle:
        ax.text(
            0.0, 1.0, subtitle, transform=ax.transAxes,
            fontsize=8.5, color=TEXT_SECONDARY, va="bottom",
            # Nudge clear of the axes line without reaching the title baseline.
            bbox=dict(facecolor="none", edgecolor="none", pad=3.0),
        )
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
