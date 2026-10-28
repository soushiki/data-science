"""Export a compact dashboard.json (< 500 KB) for the static site."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import DASHBOARD, METRICS, PREDS, QUALITY, SEED
from src.train import FEATURES_PQ

WEEKS = ["2025-07-14", "2025-10-13", "2025-12-22", "2026-01-12", "2026-04-13", "2026-07-13"]
SEASONS = {"Winter": [12, 1, 2], "Spring": [3, 4, 5], "Summer": [6, 7, 8], "Autumn": [9, 10, 11]}


def main() -> None:
    m = json.loads(METRICS.read_text())
    q = json.loads(QUALITY.read_text())
    pred = pd.read_parquet(PREDS)
    daily = pd.read_parquet(PREDS.with_name("test_daily.parquet"))
    df = pd.read_parquet(FEATURES_PQ)
    p, itv = m["peak"], m["interval"]
    r0 = lambda x: int(round(x))

    kpis = {
        "test_start": m["splits"]["test"][0], "test_end": m["splits"]["test"][1],
        "test_days": p["n_days"], "test_half_hours": m["splits"]["test"][2],
        "model_mae_mw": r0(m["test"]["hgb"]["mae"]), "model_mape": round(m["test"]["hgb"]["mape"], 4),
        "naive_mae_mw": r0(m["test"]["naive"]["mae"]),
        "peak_mae_model_mw": r0(p["hgb"]["mae"]), "peak_mae_neso_mw": r0(p["neso"]["mae"]),
        "peak_mape_model": round(p["hgb"]["mape"], 4), "peak_mape_neso": round(p["neso"]["mape"], 4),
        "pct_days_model_wins": round(p["pct_days_model_beats_neso"], 3),
        "interval_coverage_80": round(itv["test_calibrated_coverage"], 3),
        "last_demand_date": q["demand"]["last_date"],
    }
    table = lambda split: [{"model": k, "mae": r0(v["mae"]), "mape": round(v["mape"], 4),
                            "rmse": r0(v["rmse"]), "bias": r0(v["bias"])} for k, v in m[split].items()]
    peak_tbl = [{"model": k, "mae": r0(p[k]["mae"]), "mape": round(p[k]["mape"], 4),
                 "rmse": r0(p[k]["rmse"]), "bias": r0(p[k]["bias"])}
                for k in ("naive", "ridge", "hgb", "hgb_bc", "neso", "neso_bc", "blend")]
    h2h = {"n_days": p["n_days"], "tables": peak_tbl, "val_bias_neso_mw": r0(p["val_bias_neso"]),
           "pct_days_model_wins": round(p["pct_days_model_beats_neso"], 3),
           "pct_days_model_wins_bias_corrected": round(p["pct_days_model_bc_beats_neso_bc"], 3),
           "dp_only": {k: (v if not isinstance(v, dict) else {a: round(b, 4) for a, b in v.items()})
                       for k, v in p["dp_only"].items()},
           "trough": {k: (v if not isinstance(v, dict) else {a: round(b, 4) for a, b in v.items()})
                      for k, v in p["trough"].items()},
           "by_daytype": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
                          for r in p["by_daytype"]]}
    monthly = [{"month": r["month"], "n": r["n"], "model": r0(r["model_mae"]), "neso": r0(r["neso_mae"]),
                "model_bias": r0(r["model_bias"]), "neso_bias": r0(r["neso_bias"])} for r in p["by_month"]]

    dseries = daily.dropna(subset=["neso_peak"])
    daily_series = {"date": [d.strftime("%Y-%m-%d") for d in dseries["date"]],
                    "actual": dseries["act_peak"].round().astype(int).tolist(),
                    "model": dseries["hgb_peak"].round().astype(int).tolist(),
                    "neso": dseries["neso_peak"].round().astype(int).tolist()}

    weeks = []
    for w in WEEKS:
        s = pred[(pred["date"] >= w) & (pred["date"] < pd.Timestamp(w) + pd.Timedelta(days=7))]
        assert len(s) == 336, (w, len(s))
        weeks.append({"start": w, "step_min": 30,
                      "actual": s["y"].astype(int).tolist(), "pred": s["hgb"].round().astype(int).tolist(),
                      "p10": s["p10c"].round().astype(int).tolist(), "p90": s["p90c"].round().astype(int).tolist()})

    d = df.groupby("date").agg(nd=("ND", "mean"), t=("temp", "mean"), dow=("dow", "first"),
                               hol=("holiday", "first"), n=("ND", "size"), year=("trend", "first"))
    d = d[d["n"] == 48].sample(1000, random_state=SEED).sort_index()
    scatter = {"t": d["t"].round(1).tolist(), "nd": d["nd"].round().astype(int).tolist(),
               "working": ((d["dow"] < 5) & (d["hol"] == 0)).astype(int).tolist()}

    wd = df[(df["dow"] < 5) & (df["holiday"] == 0)]
    profiles = {k: wd[wd["month"].isin(v)].groupby("slot")["ND"].mean().round().astype(int).tolist()
                for k, v in SEASONS.items()}
    s = wd[wd["month"].isin([6, 7, 8])].assign(year=lambda x: x["date"].dt.year)
    duck = {"profiles": {str(y): g.groupby("slot")["ND"].mean().round().astype(int).tolist()
                         for y, g in s.groupby("year")},
            "midday_mw": {str(y): r0(g[(g["slot"] >= 24) & (g["slot"] < 28)]["ND"].mean()) for y, g in s.groupby("year")},
            "evening_mw": {str(y): r0(g[(g["slot"] >= 36) & (g["slot"] < 40)]["ND"].mean()) for y, g in s.groupby("year")},
            "solar_capacity_mw": {str(y): r0(g["solar_cap"].mean()) for y, g in s.groupby("year")}}

    imp = pd.Series(m["importance"]).sort_values(ascending=False).head(12)
    out = {
        "question": "Can a weather-driven model beat Great Britain's grid operator at forecasting tomorrow's electricity demand?",
        "kpis": kpis, "splits": m["splits"],
        "metrics": {"test": table("test"), "val": table("val")},
        "head_to_head": h2h, "monthly_peak_mae": monthly, "daily_peaks": daily_series,
        "weeks": weeks, "temp_scatter": scatter, "season_profiles": profiles, "duck": duck,
        "importance": [{"feature": k, "mae_increase_mw": round(v, 1)} for k, v in imp.items()],
        "interval": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in itv.items()
                     if k != "calibrated_coverage_by_month"},
        "hyperparameters": m["hyperparameters"],
        "data": {"demand_first": q["demand"]["first_timestamp_utc"][:10], "demand_last": q["demand"]["last_date"],
                 "demand_missing_half_hours": q["demand"]["missing_half_hours"],
                 "weather_source_share": q["weather"]["source_share"],
                 "neso_days": q["neso_forecast"]["days"], "neso_days_with_dp": q["neso_forecast"]["days_with_dp"],
                 "neso_cardinal_points": q["neso_forecast"]["cardinal_points"]},
    }
    DASHBOARD.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"dashboard.json: {DASHBOARD.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
