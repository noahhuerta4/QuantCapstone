"""Assemble the write-up, ``Report/Market-Neutral-Reversal-Book.html``.

The source lives in ``Report/src``: a stylesheet and one HTML fragment per
section, with figures referenced as ``__IMG_<name>__`` placeholders. Building
inlines ``Report/figures/<name>.png`` as base64 so the result is one
self-contained file that can be emailed or opened offline.

The figures come from ``quant-capstone study figures`` and the numbers in the
prose come from ``results/``; ``tests/test_docs.py`` checks the two agree.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

ROOT = Path("Report")
OUTPUT = ROOT / "Market-Neutral-Reversal-Book.html"
PLACEHOLDER = re.compile(r"__IMG_([a-z_]+)__")

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Market-Neutral Reversal Book</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&display=swap">
<style>
{css}</style>
</head>
<body>
"""


def sections(root: Path = ROOT) -> list[Path]:
    return sorted((root / "src" / "sections").glob("*.html"))


def body(root: Path = ROOT) -> str:
    """Masthead, then every numbered section inside the page wrapper."""
    parts = [p.read_text(encoding="utf-8") for p in sections(root)]
    return parts[0] + '\n<div class="wrap">\n\n' + "\n".join(parts[1:]) + "\n</div>\n"


def _inline(html: str, figures: Path) -> str:
    def encode(match: re.Match) -> str:
        path = figures / f"{match.group(1)}.png"
        if not path.exists():
            raise FileNotFoundError(f"{path} -- run `quant-capstone study figures` first")
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()

    return PLACEHOLDER.sub(encode, html)


def build(root: Path = ROOT, out: Path = OUTPUT) -> Path:
    css = (root / "src" / "style.css").read_text(encoding="utf-8")
    html = HEAD.format(css=css) + _inline(body(root), root / "figures") + "\n</body>\n</html>\n"
    out.write_text(html, encoding="utf-8")
    return out


def text(root: Path = ROOT) -> str:
    """The prose with tags stripped, for checking the numbers it quotes."""
    import html as html_lib

    raw = re.sub(r"<[^>]+>", " ", body(root))
    return re.sub(r"\s+", " ", html_lib.unescape(raw))
