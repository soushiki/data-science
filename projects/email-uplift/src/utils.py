"""Small helpers shared by pipeline stages."""
from __future__ import annotations

import json

import numpy as np

from src.config import METRICS


class Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_metrics(section: str, obj) -> None:
    """Merge one section into reports/metrics.json."""
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    m = json.loads(METRICS.read_text()) if METRICS.exists() else {}
    m[section] = obj
    METRICS.write_text(json.dumps(m, indent=1, cls=Enc))


def load_metrics() -> dict:
    return json.loads(METRICS.read_text())
