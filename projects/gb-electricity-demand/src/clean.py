"""Clean raw data: tz-aware half-hourly demand, half-hourly weather, daily NESO forecasts.

Settlement periods are numbered from local midnight (period 1 starts 00:00 local),
so timestamp = local midnight (Europe/London) -> UTC + (period-1) * 30 min. This
handles the 46- and 50-period clock-change days without special-casing.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import (CLEAN, DEMAND_RAW, NESO_CLEAN, NESO_RAW, QUALITY, WEATHER_CLEAN,
                        WEATHER_RAW, WEIGHTS)

TZ = "Europe/London"
KEEP = ["ND", "TSD", "EMBEDDED_SOLAR_GENERATION", "EMBEDDED_SOLAR_CAPACITY",
        "EMBEDDED_WIND_GENERATION", "EMBEDDED_WIND_CAPACITY"]


def parse_dates(s: pd.Series) -> pd.Series:
    """The NESO files mix '2024-06-03', '30-DEC-2022' and '31-Oct-23' formats."""
    s = s.str.title()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[us]")
    iso = s.str.match(r"\d{4}-\d{2}-\d{2}$")
    out[iso] = pd.to_datetime(s[iso], format="%Y-%m-%d")
    for fmt in ("%d-%b-%Y", "%d-%b-%y"):
        rest = out.isna()
        out[rest] = pd.to_datetime(s[rest], format=fmt, errors="coerce")
    return out


def clean_demand() -> tuple[pd.DataFrame, dict]:
    d = pd.read_parquet(DEMAND_RAW)
    q: dict = {"raw_rows": len(d)}
    d["date"] = parse_dates(d["SETTLEMENT_DATE"])
    q["unparseable_dates"] = int(d["date"].isna().sum())
    d = d.dropna(subset=["date"])
    q["duplicate_rows_dropped"] = int(d.duplicated(["date", "SETTLEMENT_PERIOD"]).sum())
    d = d.drop_duplicates(["date", "SETTLEMENT_PERIOD"], keep="last")
    d = d.sort_values(["date", "SETTLEMENT_PERIOD"])
    midnight = d["date"].dt.tz_localize(TZ).dt.tz_convert("UTC")
    d["ts"] = midnight + pd.to_timedelta((d["SETTLEMENT_PERIOD"] - 1) * 30, unit="min")
    d = d.drop_duplicates("ts")
    # expected number of periods per local day (23/24/25 hours)
    days = pd.DatetimeIndex(d["date"].unique())
    start = days.tz_localize(TZ).tz_convert("UTC")
    end = (days + pd.Timedelta(days=1)).tz_localize(TZ).tz_convert("UTC")
    expected = pd.Series(((end - start) / pd.Timedelta(minutes=30)).astype(int), index=days)
    cnt = d.groupby("date").size()
    bad = cnt[cnt != expected.reindex(cnt.index)]
    q["days"] = int(len(cnt))
    q["days_with_wrong_period_count"] = {str(k.date()): int(v) for k, v in bad.items()}
    q["dst_days_46_periods"] = int((expected == 46).sum())
    q["dst_days_50_periods"] = int((expected == 50).sum())
    q["period_range"] = [int(d["SETTLEMENT_PERIOD"].min()), int(d["SETTLEMENT_PERIOD"].max())]
    full = pd.date_range(d["ts"].min(), d["ts"].max(), freq="30min")
    q["missing_half_hours"] = int(len(full.difference(pd.DatetimeIndex(d["ts"]))))
    q["first_timestamp_utc"], q["last_timestamp_utc"] = str(d["ts"].min()), str(d["ts"].max())
    q["last_date"] = str(d["date"].max().date())
    q["nd_nulls"] = int(d["ND"].isna().sum())
    q["nd_min"], q["nd_max"] = int(d["ND"].min()), int(d["ND"].max())
    q["nd_nonpositive"] = int((d["ND"] <= 0).sum())
    # keep only complete trailing days
    last_cnt, last = cnt.iloc[-1], cnt.index[-1]
    if last_cnt != expected[last]:
        d = d[d["date"] < last]
        q["partial_last_day_dropped"] = str(last.date())
    local = d["ts"].dt.tz_convert(TZ)
    d["slot"] = local.dt.hour * 2 + local.dt.minute // 30  # local clock half-hour 0..47
    out = d[["ts", "date", "SETTLEMENT_PERIOD", "slot"] + KEEP].rename(
        columns={"SETTLEMENT_PERIOD": "period", "EMBEDDED_SOLAR_GENERATION": "solar_gen",
                 "EMBEDDED_SOLAR_CAPACITY": "solar_cap", "EMBEDDED_WIND_GENERATION": "wind_gen",
                 "EMBEDDED_WIND_CAPACITY": "wind_cap"}).reset_index(drop=True)
    return out, q


def clean_weather() -> tuple[pd.DataFrame, dict]:
    w = pd.read_parquet(WEATHER_RAW)
    q = {"source_share": w.groupby("source").size().div(len(w)).round(4).to_dict()}
    by_year = w.assign(year=w["time"].dt.year).groupby(["year", "source"]).size().unstack(fill_value=0)
    q["rows_by_year_source"] = {str(k): v for k, v in by_year.to_dict("index").items()}
    cols = ["temperature_2m", "shortwave_radiation", "wind_speed_10m", "cloud_cover"]
    q["null_share"] = w[cols].isna().mean().round(5).to_dict()
    w["wt"] = w["city"].map(WEIGHTS)
    parts = {c: (w[c] * w["wt"]).groupby(w["time"]).sum() / w["wt"].groupby(w["time"]).sum()
             for c in cols}
    h = pd.DataFrame(parts).sort_index()
    h = h[~h.index.duplicated()]
    half = h.resample("30min").interpolate("linear")
    half = half.reindex(pd.date_range(h.index.min(), h.index.max() + pd.Timedelta(minutes=30),
                                      freq="30min")).ffill()
    half.index.name = "ts"
    return half.reset_index(), q


def clean_neso() -> tuple[pd.DataFrame, dict]:
    n = pd.read_parquet(NESO_RAW)
    n = n.sort_values("FORECAST_TIMESTAMP")
    q = {"raw_rows": len(n), "cardinal_points": sorted(n["CARDINALPOINT"].unique())}
    # several re-issues per day (08:xx and ~11:45): keep the first issue = the 08:45-ish D-1 one
    n = n.groupby(["TARGETDATE", "CARDINALPOINT", "CP_ST_TIME", "CP_END_TIME"], as_index=False).first()
    q["rows_after_first_issue"] = len(n)
    g = n.groupby("TARGETDATE")["FORECASTDEMAND"]
    daily = pd.DataFrame({"neso_peak": g.max(), "neso_min": g.min(), "n_points": g.size()})
    dp = n[n["CARDINALPOINT"] == "DP"].groupby("TARGETDATE")["FORECASTDEMAND"].max()
    daily["neso_dp"] = dp
    daily.index.name = "date"
    q["days"] = int(len(daily))
    q["days_with_dp"] = int(daily["neso_dp"].notna().sum())
    ts = pd.to_datetime(n["FORECAST_TIMESTAMP"])
    secs = ts.dt.hour * 3600 + ts.dt.minute * 60
    q["median_issue_clock_time"] = str(pd.Timedelta(seconds=float(secs.median())))
    return daily.reset_index(), q


def main() -> None:
    dem, q1 = clean_demand()
    w, q2 = clean_weather()
    nes, q3 = clean_neso()
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    dem.to_parquet(CLEAN, index=False)
    w.to_parquet(WEATHER_CLEAN, index=False)
    nes.to_parquet(NESO_CLEAN, index=False)
    # weather coverage vs demand
    m = dem.merge(w, on="ts", how="left")
    q2["demand_rows_without_weather"] = int(m["temperature_2m"].isna().sum())
    QUALITY.parent.mkdir(parents=True, exist_ok=True)
    QUALITY.write_text(json.dumps({"demand": q1, "weather": q2, "neso_forecast": q3}, indent=2, default=str))
    print(json.dumps({"demand": q1, "weather": q2, "neso": q3}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
