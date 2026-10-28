"""Validate and clean the raw CSV; write clean.parquet and reports/data_quality.json."""
from __future__ import annotations

import json
import re

import pandas as pd

from src.config import ARMS, CLEAN, QUALITY, RAW_CSV
from src.utils import Enc

ARM_CODE = {v: k for k, v in ARMS.items()}


def main() -> pd.DataFrame:
    raw = pd.read_csv(RAW_CSV)
    q: dict = {"n_rows": len(raw), "n_cols": raw.shape[1],
               "nulls": raw.isna().sum().to_dict(),
               "full_row_duplicates": int(raw.duplicated().sum()),
               "note_duplicates": "No customer id exists, so identical rows cannot be told apart "
                                  "from distinct customers with identical attributes; kept."}
    df = raw.copy()
    q["zip_code_raw_values"] = sorted(df["zip_code"].unique())
    df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})   # typo in source file
    q["segment_values"] = df["segment"].value_counts().to_dict()
    df["arm_code"] = df["segment"].map(ARM_CODE).astype(int)
    df["arm"] = df["arm_code"].map(ARMS)

    # integrity checks
    q["checks"] = {
        "conversion_without_visit": int(((df["conversion"] == 1) & (df["visit"] == 0)).sum()),
        "spend_positive_without_conversion": int(((df["spend"] > 0) & (df["conversion"] == 0)).sum()),
        "conversion_with_zero_spend": int(((df["conversion"] == 1) & (df["spend"] == 0)).sum()),
        "negative_spend": int((df["spend"] < 0).sum()),
        "negative_history": int((df["history"] < 0).sum()),
        "recency_range": [int(df["recency"].min()), int(df["recency"].max())],
        "history_range": [float(df["history"].min()), float(df["history"].max())],
        "binary_cols_valid": bool(all(df[c].isin([0, 1]).all() for c in
                                      ["mens", "womens", "newbie", "visit", "conversion"])),
        "no_mens_no_womens": int(((df["mens"] == 0) & (df["womens"] == 0)).sum()),
    }
    # history_segment bands must agree with the numeric history
    bands = df.groupby("history_segment")["history"].agg(["min", "max", "size"])
    q["history_segment_bands"] = {k: {"min": float(r["min"]), "max": float(r["max"]), "n": int(r["size"])}
                                  for k, r in bands.iterrows()}
    mismatches = 0
    for seg in bands.index:
        nums = [float(x.replace(",", "")) for x in re.findall(r"\$([\d,]+)", seg)]
        lo = nums[0]
        hi = float("inf") if ("+" in seg or len(nums) == 1) else nums[1]
        h = df.loc[df["history_segment"] == seg, "history"]
        mismatches += int(((h < lo - 0.01) | (h > hi + 0.01)).sum())
    q["checks"]["history_outside_segment_band"] = mismatches
    q["spend_when_converted"] = df.loc[df["conversion"] == 1, "spend"].describe().round(2).to_dict()
    q["spend_zero_share"] = float((df["spend"] == 0).mean())

    # modelling features
    df["treat"] = (df["arm_code"] > 0).astype(int)
    df["zip_suburban"] = (df["zip_code"] == "Suburban").astype(int)
    df["zip_urban"] = (df["zip_code"] == "Urban").astype(int)
    df["chan_phone"] = (df["channel"] == "Phone").astype(int)
    df["chan_web"] = (df["channel"] == "Web").astype(int)
    df = df.drop(columns=["segment"])
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN, index=False)
    QUALITY.parent.mkdir(parents=True, exist_ok=True)
    QUALITY.write_text(json.dumps(q, indent=1, cls=Enc))
    print(f"clean rows={len(df)} dupes={q['full_row_duplicates']} checks={q['checks']}")
    return df


if __name__ == "__main__":
    main()
