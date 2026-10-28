"""Export small aggregates for the web dashboard (< 500 KB)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.config import CLEAN, DASHBOARD, METRICS, PREDS, PREDS2, SEED, THRESHOLDS2


def _r(x: float, n: int = 4) -> float | None:
    return None if pd.isna(x) else round(float(x), n)


def agg(g) -> pd.DataFrame:
    """Count, missed rate and median resolution hours per group."""
    return g.agg(n=("missed_72h", "size"), missed_rate=("missed_72h", "mean"),
                 median_hours=("resolution_hours", "median")).reset_index()


def records(t: pd.DataFrame) -> list[dict]:
    out = []
    for r in t.to_dict("records"):
        row = {}
        for k, v in r.items():
            if isinstance(v, str):
                row[k] = v
            elif k == "n":
                row[k] = int(v)
            else:
                row[k] = _r(v, 2 if "hours" in k else 4)
        out.append(row)
    return out


def pr_points(y: np.ndarray, s: np.ndarray, n: int = 100) -> list[list[float]]:
    """Downsampled (recall, precision) pairs."""
    p, r, _ = precision_recall_curve(y, s)
    idx = np.unique(np.linspace(0, len(p) - 1, n).astype(int))
    return [[round(float(r[i]), 4), round(float(p[i]), 4)] for i in idx]


def main() -> None:
    df = pd.read_parquet(CLEAN)
    pred = pd.read_parquet(PREDS)
    allm = json.loads(METRICS.read_text())
    m, m2 = allm["study1_72h"], allm["study2_delay_vs_peers"]
    pred2 = pd.read_parquet(PREDS2)
    thr = pd.read_parquet(THRESHOLDS2).set_index("complaint_type")["p75_hours"]
    df["month"] = df["created_date"].dt.strftime("%Y-%m")
    top = df["complaint_type"].value_counts().head(20).index
    grid = (df.assign(d=df["created_date"].dt.weekday, h=df["created_date"].dt.hour)
              .groupby(["d", "h"])["missed_72h"].agg(["mean", "size"]).reset_index())
    pts = df.dropna(subset=["latitude"]).sample(300, random_state=SEED)
    out = {
        "kpis": {"n_requests": int(len(df)), "base_missed_rate": _r(df["missed_72h"].mean()),
                 "median_resolution_hours": _r(df["resolution_hours"].median(), 2),
                 "share_still_open": _r(df["is_open"].mean()),
                 "date_range": [str(df["created_date"].min().date()),
                                str(df["created_date"].max().date())],
                 "note": "~20% systematic sample (unique_key ending 3 or 7) of NYC 311"},
        "monthly": records(agg(df.groupby("month"))),
        "by_complaint_type": records(agg(df[df["complaint_type"].isin(top)]
                                         .groupby("complaint_type")).sort_values("n", ascending=False)),
        "by_borough": records(agg(df[df["borough"] != "Unknown"].groupby("borough"))),
        "hour_weekday": [{"weekday": int(r.d), "hour": int(r.h), "missed_rate": _r(r["mean"]),
                          "n": int(r["size"])} for _, r in grid.iterrows()],
        "model_comparison": {"best_model": m["best_model"], "test": m["test"], "val": m["val"],
                             "base_rate_test": m["base_rate"]["test"]},
        "pr_curves": {k: pr_points(pred["y"].to_numpy(), pred[k].to_numpy())
                      for k in m["test"]},
        "feature_importance": [{"feature": d["feature"], "importance": _r(d["importance"], 5)}
                               for d in m["permutation_importance"][:12]],
        "study2": {
            "description": "Delayed vs peers: slower than the train p75 for the same complaint type",
            "model_comparison": {"best_model": m2["best_model"], "test": m2["test"],
                                 "val": m2["val"], "base_rate_test": m2["base_rate"]["test"]},
            "pr_curves": {k: pr_points(pred2["y"].to_numpy(), pred2[k].to_numpy())
                          for k in m2["test"]},
            "gains_curves": m2["gains_curves"],
            "feature_importance": [{"feature": d["feature"], "importance": _r(d["importance"], 5)}
                                   for d in m2["permutation_importance"][:12]],
            "p75_threshold_hours": {t: _r(thr.get(t, thr.get("Other")), 2) for t in top},
        },
        "map_points": [[round(a, 4), round(b, 4), int(c)] for a, b, c in
                       zip(pts["latitude"], pts["longitude"], pts["missed_72h"])],
    }
    DASHBOARD.write_text(json.dumps(out, separators=(",", ":")))
    print(f"dashboard.json: {DASHBOARD.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
