"""Inline dashboard.json into site/template.html -> docs/index.html (GitHub Pages)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "site" / "template.html"
DATA = ROOT / "data" / "processed" / "dashboard.json"
# GitHub Pages serves the repo-level docs/ folder; each project gets a subfolder.
OUT = ROOT.parents[1] / "docs" / ROOT.name / "index.html"


def build() -> Path:
    """Write a single self-contained HTML page with the dashboard data embedded."""
    data = json.loads(DATA.read_text(encoding="utf-8"))
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(data, separators=(",", ":"))
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    print(f"wrote {build()}")
