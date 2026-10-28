"""Download NESO demand + day-ahead forecast archive and Open-Meteo weather.

Cached to data/raw/; skipped if present unless --refresh.
Weather prefers the Open-Meteo *historical forecast* API (archived NWP forecasts,
what was knowable on D-1). Where it has no data, falls back to the observation
reanalysis archive; the `source` column records which (fallback = optimistic).
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from src.config import (API, ARCHIVE_API, CITIES, DEMAND_RAW, DEMAND_RESOURCES,
                        HIST_FORECAST_API, NESO_FORECAST_RESOURCE, NESO_RAW, RAW, START,
                        TODAY, WEATHER_RAW, WEATHER_VARS)


def _get_json(url: str, retries: int = 6) -> dict:
    """GET JSON with exponential backoff."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gb-demand-portfolio/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 400 and "open-meteo" in url:
                raise
            wait = 2 ** attempt * 2
            print(f"  request failed ({e}); retry in {wait}s")
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            wait = 2 ** attempt * 2
            print(f"  request failed ({e}); retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"giving up on {url}")


def _datastore(resource: str, filters: dict | None = None, limit: int = 32000) -> pd.DataFrame:
    rows, offset = [], 0
    while True:
        q = {"resource_id": resource, "limit": limit, "offset": offset}
        if filters:
            q["filters"] = json.dumps(filters)
        res = _get_json(f"{API}?{urllib.parse.urlencode(q)}")["result"]
        rows += res["records"]
        offset += limit
        if len(res["records"]) < limit:
            break
    df = pd.DataFrame(rows)
    return df.drop(columns=[c for c in ("_id", "_full_text") if c in df.columns])


def fetch_demand() -> pd.DataFrame:
    parts = []
    for yr, rid in DEMAND_RESOURCES.items():
        d = _datastore(rid)
        print(f"  demand {yr}: {len(d):,} rows")
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def fetch_neso() -> pd.DataFrame:
    d = _datastore(NESO_FORECAST_RESOURCE)
    print(f"  neso forecast archive (all): {len(d):,} rows")
    d["TARGETDATE"] = pd.to_datetime(d["TARGETDATE"], errors="coerce")
    d["DAYSAHEAD"] = pd.to_numeric(d["DAYSAHEAD"], errors="coerce")
    d = d[(d["TARGETDATE"] >= START) & (d["DAYSAHEAD"] == 1)]
    print(f"  after filter (>= {START}, DAYSAHEAD=1): {len(d):,} rows")
    return d.reset_index(drop=True)


def _weather_chunk(base: str, lat: float, lon: float, s: str, e: str) -> pd.DataFrame | None:
    q = {"latitude": lat, "longitude": lon, "start_date": s, "end_date": e,
         "hourly": ",".join(WEATHER_VARS), "timezone": "UTC"}
    try:
        h = _get_json(f"{base}?{urllib.parse.urlencode(q)}")["hourly"]
    except urllib.error.HTTPError:
        return None
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_weather() -> pd.DataFrame:
    """Per-city hourly weather in *UTC*: archived forecast where available else observed."""
    frames = []
    ranges = [(f"{y}-01-01", f"{y}-12-31" if y < 2026 else TODAY) for y in range(2022, 2027)]
    for city, (lat, lon, _) in CITIES.items():
        for s, e in ranges:
            for base, src in ((HIST_FORECAST_API, "forecast"), (ARCHIVE_API, "observed")):
                d = _weather_chunk(base, lat, lon, s, e)
                time.sleep(0.5)
                if d is not None and d["temperature_2m"].notna().mean() > 0.5:
                    d["city"], d["source"] = city, src
                    frames.append(d)
                    print(f"  weather {city} {s[:4]}: {src} ({d['temperature_2m'].notna().mean():.0%} filled)")
                    break
            else:
                print(f"  weather {city} {s[:4]}: NO DATA")
    w = pd.concat(frames, ignore_index=True)
    w["time"] = w["time"].dt.tz_localize("UTC")
    return w


def main(refresh: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for path, fn in ((DEMAND_RAW, fetch_demand), (NESO_RAW, fetch_neso), (WEATHER_RAW, fetch_weather)):
        if path.exists() and not refresh:
            print(f"skip {path.name} (cached)")
            continue
        print(f"fetching {path.name}")
        df = fn()
        df = df.astype({c: str for c in df.columns if df[c].dtype == object})
        df.to_parquet(path, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    main(ap.parse_args().refresh)
