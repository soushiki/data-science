"""Clean raw 311 data, build the target `missed_72h`, log data-quality stats."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import CLEAN, QUALITY, RAW, THRESHOLD_H

NYC_LAT, NYC_LON = (40.45, 40.95), (-74.30, -73.65)
BOROUGHS = {"MANHATTAN": "Manhattan", "BROOKLYN": "Brooklyn", "QUEENS": "Queens",
            "BRONX": "Bronx", "STATEN ISLAND": "Staten Island"}


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return cleaned frame and a data-quality log (rows dropped per rule)."""
    q: dict = {"rows_raw": int(len(df)), "dropped": {}, "flagged": {}}
    df = df.copy()

    n = len(df)
    df = df.drop_duplicates("unique_key")
    q["dropped"]["duplicate_unique_key"] = n - len(df)

    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce")

    n = len(df)
    df = df[df["created_date"].notna()]
    q["dropped"]["missing_created_date"] = n - len(df)

    n = len(df)
    df = df[~(df["closed_date"] < df["created_date"])]
    q["dropped"]["closed_before_created"] = n - len(df)

    n = len(df)
    df = df[~(df["closed_date"] > pd.Timestamp("2026-10-28"))]
    q["dropped"]["closed_in_future"] = n - len(df)

    # Suspicious closures: kept but flagged (they are plausible auto-closures).
    dur = df["closed_date"] - df["created_date"]
    df["closed_instant"] = (dur == pd.Timedelta(0)).fillna(False)
    cd = df["closed_date"]
    df["closed_midnight"] = (cd.notna() & (cd == cd.dt.normalize())).fillna(False)
    q["flagged"]["closed_within_0s"] = int(df["closed_instant"].sum())
    q["flagged"]["closed_exactly_midnight"] = int(df["closed_midnight"].sum())
    q["flagged"]["still_open"] = int(df["closed_date"].isna().sum())

    df["resolution_hours"] = (df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600
    df["is_open"] = df["closed_date"].isna()
    # Open after >=55 days counts as missed.
    df["missed_72h"] = (df["is_open"] | (df["resolution_hours"] > THRESHOLD_H)).astype(int)

    b = df["borough"].astype("string").str.upper().str.strip()
    df["borough"] = b.map(BOROUGHS).fillna("Unknown")
    q["flagged"]["borough_unspecified"] = int((df["borough"] == "Unknown").sum())
    z = df["incident_zip"].astype("string").str.extract(r"^(\d{5})")[0]
    q["flagged"]["zip_invalid_or_missing"] = int(z.isna().sum())
    df["incident_zip"] = z

    for c in ("latitude", "longitude"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = ~df["latitude"].between(*NYC_LAT) | ~df["longitude"].between(*NYC_LON)
    q["flagged"]["latlon_missing_or_outside_nyc"] = int(bad.sum())
    df.loc[bad, ["latitude", "longitude"]] = np.nan

    for c in ("agency", "complaint_type", "descriptor", "location_type", "open_data_channel_type"):
        df[c] = df[c].astype("string").str.strip().fillna("Unknown")

    df = df.sort_values(["created_date", "unique_key"]).reset_index(drop=True)
    q["rows_clean"] = int(len(df))
    q["base_missed_rate"] = float(df["missed_72h"].mean())
    q["median_resolution_hours"] = float(df["resolution_hours"].median())
    return df, q


def main() -> None:
    df, q = clean(pd.read_parquet(RAW))
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    QUALITY.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN, index=False)
    QUALITY.write_text(json.dumps(q, indent=2))
    print(json.dumps(q, indent=2))


if __name__ == "__main__":
    main()
