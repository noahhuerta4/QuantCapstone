"""Chart palette and matplotlib defaults.

Plots use stock matplotlib (default fonts, default chrome) with a two-color
palette: navy for the primary series and dark red for the contrasting one.
Anything a chart wants to say beyond its title and axis labels belongs in the
figure caption, not in a subtitle or on-plot annotation.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

NAVY = "#1f3a68"
DARK_RED = "#8b1a1a"
GRAY = "#6e6e6e"
LIGHT_GRAY = "#d9d9d9"

SERIES = [NAVY, DARK_RED, GRAY]
"""Categorical slots 1-3, in fixed order."""

DIVERGING = LinearSegmentedColormap.from_list(
    "navy_white_darkred", [NAVY, "#ffffff", DARK_RED]
)

RC = {
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


def use_style() -> None:
    plt.rcdefaults()
    plt.rcParams.update(RC)


def finish(ax, title: str, ylabel: str = "") -> None:
    """Plain title and optional y label."""
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
