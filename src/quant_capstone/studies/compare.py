"""Diff regenerated study tables against a reference set.

Numbers are compared column by column on the columns both files share, so a
study that gained a column still checks the ones it already had. A table
matches when every shared numeric cell agrees to ``atol`` or ``rtol``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame


def compare_tables(new: Path, ref: Path, rtol: float = 1e-6, atol: float = 1e-9) -> dict:
    a, b = _read(new), _read(ref)
    shared = [c for c in a.columns if c in b.columns]
    row = {"table": new.stem, "rows": f"{len(a)}/{len(b)}", "shared_cols": len(shared),
           "max_abs_diff": 0.0, "max_rel_diff": 0.0, "status": "match"}
    if len(a) != len(b):
        row["status"] = "shape differs"
        return row
    for col in shared:
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            x, y = x.to_numpy(float), y.to_numpy(float)
            both_nan = np.isnan(x) & np.isnan(y)
            diff = np.where(both_nan, 0.0, np.abs(x - y))
            if np.isnan(diff).any():
                row["status"] = f"NaN mismatch in {col}"
                continue
            rel = diff / np.maximum(np.abs(y), 1e-12)
            row["max_abs_diff"] = max(row["max_abs_diff"], float(diff.max(initial=0)))
            row["max_rel_diff"] = max(row["max_rel_diff"], float(np.where(diff > atol, rel, 0).max(initial=0)))
            if not np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True) and row["status"] == "match":
                row["status"] = f"differs ({col}...)"
        elif not x.astype(str).equals(y.astype(str)) and row["status"] == "match":
            row["status"] = f"labels differ ({col})"
    return row


def compare_dirs(new_dir: Path, ref_dir: Path, **tol) -> pd.DataFrame:
    rows = []
    for new in sorted(Path(new_dir).glob("*.csv")):
        ref = Path(ref_dir) / new.name
        if not ref.exists():
            rows.append({"table": new.stem, "status": "no reference"})
            continue
        rows.append(compare_tables(new, ref, **tol))
    return pd.DataFrame(rows)
