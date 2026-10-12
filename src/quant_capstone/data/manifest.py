"""A fingerprint of the cached inputs every committed number was computed from.

``data_cache/`` is 180 MB and git-ignored, and most of it cannot be re-fetched
byte-for-byte: yfinance revises adjusted prices, serves only currently-listed
names, and throttles into silently empty panels; the IB hourly bars need a live
TWS session. So the code alone does not pin the results -- the data has to be
pinned too.

``results/data_manifest.json`` records each cached file's SHA-256, size and, for
the price panels, shape and density. ``verify`` tells a fresh checkout whether
it is running on the same inputs: a hash mismatch means the numbers may move
for reasons no code change explains, which is the first thing to rule out (see
the poisoned-cache episode in the README).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

CACHE_DIR = Path("data_cache")
MANIFEST = Path("results/data_manifest.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _describe(path: Path) -> dict:
    entry = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    if path.name.startswith("bars_"):
        frame = pd.read_parquet(path)
        entry["rows"], entry["columns"] = frame.shape
        entry["density"] = round(float(frame.notna().to_numpy().mean()), 4)
    return entry


def build(cache_dir: Path = CACHE_DIR) -> dict[str, dict]:
    return {p.name: _describe(p) for p in sorted(Path(cache_dir).glob("*.parquet"))}


def write(cache_dir: Path = CACHE_DIR, path: Path = MANIFEST) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(cache_dir), indent=2) + "\n", encoding="utf-8")
    return path


def verify(cache_dir: Path = CACHE_DIR, path: Path = MANIFEST) -> pd.DataFrame:
    """One row per recorded file: ``ok``, ``missing`` or ``changed``."""
    recorded = json.loads(Path(path).read_text())
    rows = []
    for name, want in recorded.items():
        file = Path(cache_dir) / name
        if not file.exists():
            status = "missing"
        else:
            status = "ok" if _sha256(file) == want["sha256"] else "changed"
        rows.append({"file": name, "status": status, "bytes": want["bytes"]})
    return pd.DataFrame(rows)
