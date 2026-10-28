"""Download the Hillstrom MineThatData e-mail experiment CSV (cached in data/raw/)."""
from __future__ import annotations

import time
import urllib.error
import urllib.request

from src.config import RAW, RAW_CSV, URL


def main(refresh: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists() and not refresh:
        print(f"cached: {RAW_CSV} ({RAW_CSV.stat().st_size / 1e6:.1f} MB)")
        return
    for attempt in range(5):
        try:
            req = urllib.request.Request(URL, headers={"User-Agent": "email-uplift-portfolio/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                RAW_CSV.write_bytes(r.read())
            print(f"downloaded {RAW_CSV.stat().st_size / 1e6:.1f} MB")
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            wait = 2 ** attempt * 2
            print(f"  request failed ({e}); retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"giving up on {URL}")


if __name__ == "__main__":
    main()
