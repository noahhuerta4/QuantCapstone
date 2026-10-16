"""Run studies by name and write their tables to ``results/studies``.

Each study module exposes ``DESCRIPTION`` and ``run() -> {name: DataFrame}``. The
runner owns the writing, so every study's output lands in the same place in the
same format, and ``check`` can regenerate everything into a scratch directory
and diff it against what is committed.
"""

from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

STUDIES_DIR = Path("results/studies")

#: Run order matters only for ``figures``, which reads three earlier outputs.
ORDER = (
    "turnover", "liquidity", "sleeves", "blends", "retune", "breadth",
    "fundamentals", "transfer", "pead", "intraday", "sleeve_survey",
    "events", "passive", "regime", "band", "leverage", "progression", "bayes", "figures",
)


def module(name: str):
    if name not in ORDER:
        raise KeyError(f"unknown study {name!r}; choose from {', '.join(ORDER)}")
    return importlib.import_module(f"{__package__}.{name}")


def _writes_index(frame: pd.DataFrame) -> bool:
    """Labelled indexes (dates, deciles, sleeve names) are data; integer ones are not."""
    return frame.index.name is not None or not pd.api.types.is_integer_dtype(frame.index)


def write(tables: dict[str, pd.DataFrame], outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in tables.items():
        path = outdir / f"{name}.csv"
        frame.to_csv(path, index=_writes_index(frame))
        paths.append(path)
    return paths


def run(names: list[str], outdir: Path = STUDIES_DIR) -> list[Path]:
    paths = []
    for name in names:
        mod = module(name)
        log.warning("study %-14s %s", name, mod.DESCRIPTION)
        start = time.time()
        tables = mod.run(outdir) if name == "figures" else mod.run()
        paths += write(tables, outdir)
        log.warning("study %-14s done in %.0fs", name, time.time() - start)
    return paths
