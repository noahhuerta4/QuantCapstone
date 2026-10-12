"""Every side study behind a number in the README or the write-up.

Before this package the studies existed only as one-off scripts in a scratch
directory, so their results could be read but not regenerated. Each module here
reproduces one study under the configuration it was originally run with (see
``eras.py``); ``uv run quant-capstone study all`` regenerates every table into
``results/studies`` and ``--check`` diffs a fresh regeneration against it.
"""

from .compare import compare_dirs
from .runner import ORDER, STUDIES_DIR, run

__all__ = ["ORDER", "STUDIES_DIR", "run", "compare_dirs"]
