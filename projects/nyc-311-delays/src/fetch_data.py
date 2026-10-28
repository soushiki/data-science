"""Download NYC 311 requests (Socrata) for 2026-03-01..2026-08-31.

Sampling: the full window holds ~1.98M requests. unique_key is a sequential id, so
its last digit is effectively a random label unrelated to complaint content or
timing. Keeping keys ending in 3 or 7 yields a ~20% systematic sample (~400k rows)
that is uniform across months, agencies and hours. Consequence: the engineered
"agency backlog" feature counts sampled requests (~20% of the true backlog); it is
a scaled proxy, which is fine for a tree model / relative ranking.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from src.config import END, RAW, START

URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
COLUMNS = ["unique_key", "created_date", "closed_date", "agency", "complaint_type",
           "descriptor", "location_type", "incident_zip", "borough",
           "open_data_channel_type", "latitude", "longitude", "status"]
PAGE = 50_000
SAMPLE_DIGITS = ("3", "7")


def _get(params: dict[str, str], retries: int = 6) -> list[dict]:
    """GET one page with exponential backoff on HTTP/network errors."""
    url = f"{URL}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            wait = 2 ** attempt * 2
            print(f"  request failed ({e}); retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Socrata request failed after retries")


def fetch() -> pd.DataFrame:
    """Page through the sampled window ordered by unique_key."""
    like = " OR ".join(f"unique_key like '%{d}'" for d in SAMPLE_DIGITS)
    where = (f"created_date >= '{START}T00:00:00' AND created_date < '{END}T00:00:00'"
             f" AND ({like})")
    frames, offset = [], 0
    while True:
        rows = _get({"$select": ",".join(COLUMNS), "$where": where,
                     "$order": "unique_key", "$limit": str(PAGE), "$offset": str(offset)})
        print(f"offset {offset}: {len(rows)} rows")
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        offset += PAGE
        if len(rows) < PAGE:
            break
    df = pd.concat(frames, ignore_index=True)
    for c in COLUMNS:
        if c not in df:
            df[c] = None
    return df[COLUMNS]


def main(refresh: bool = False) -> None:
    if RAW.exists() and not refresh:
        print(f"{RAW} exists; skipping download (use --refresh)")
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    df = fetch()
    df.to_parquet(RAW, index=False)
    print(f"saved {len(df):,} rows -> {RAW}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    main(ap.parse_args().refresh)
