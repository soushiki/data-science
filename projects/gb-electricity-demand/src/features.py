"""Feature engineering for the day-ahead (D-1, 09:00) half-hourly demand forecast.

LEAKAGE RULES (enforced in `build_features`, verified by `assert_no_leakage`)
-----------------------------------------------------------------------------
A forecast is issued at 09:00 on D-1 for all periods of day D.
  ALLOWED  : demand actuals up to the end of D-2 (conservative: publication lag),
             calendar and UK bank holidays, embedded solar *capacity* as of D-2,
             weather *forecasts* for D (and D-1, D-2 from the same forecast series).
  FORBIDDEN: any demand actual (ND, TSD, embedded generation output, flows) from D-1 or D.
All demand-derived features are built from a day-level matrix looked up at >= 2 days back;
`assert_no_leakage` multiplies every actual on D-1 and D and confirms that the features
of day D do not change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOLIDAYS = pd.to_datetime([  # England & Wales bank holidays (+ New Year / Christmas Day itself)
    "2022-01-01", "2022-01-03", "2022-04-15", "2022-04-18", "2022-05-02", "2022-06-02", "2022-06-03",
    "2022-08-29", "2022-09-19", "2022-12-25", "2022-12-26", "2022-12-27",
    "2023-01-01", "2023-01-02", "2023-04-07", "2023-04-10", "2023-05-01", "2023-05-08", "2023-05-29",
    "2023-08-28", "2023-12-25", "2023-12-26",
    "2024-01-01", "2024-03-29", "2024-04-01", "2024-05-06", "2024-05-27", "2024-08-26",
    "2024-12-25", "2024-12-26",
    "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-05", "2025-05-26", "2025-08-25",
    "2025-12-25", "2025-12-26",
    "2026-01-01", "2026-04-03", "2026-04-06", "2026-05-04", "2026-05-25", "2026-08-31",
    "2026-12-25", "2026-12-26", "2026-12-28"])
HDD_BASE, CDD_BASE = 15.5, 22.0
LAGS = ["lag_d2", "lag_d7", "lag_d14", "lag_wd4_mean", "d2_daymean", "d2_daymax", "wk_mean_d2"]
WX = ["temp", "hdd", "cdd", "rad", "wind", "cloud", "solar_pot", "temp_dmean", "temp_dmin",
      "temp_dmax", "temp_prev_dmean", "temp_ma3"]
CAL = ["slot", "dow", "month", "doy_sin", "doy_cos", "holiday", "hol_eve", "hol_after",
       "xmas_days", "xmas_window", "trend", "solar_cap_d2"]
FEATURES = CAL + WX + LAGS


def _wide(df: pd.DataFrame, col: str, dates: pd.DatetimeIndex) -> np.ndarray:
    """date x 48-slot matrix of `col` (mean over clock-change duplicates, gaps interpolated)."""
    w = df.pivot_table(index="date", columns="slot", values=col, aggfunc="mean")
    w = w.reindex(index=dates, columns=range(48)).interpolate(axis=1, limit_direction="both")
    return w.to_numpy()


def _lookup(mat: np.ndarray, day_idx: np.ndarray, slot: np.ndarray, k: int) -> np.ndarray:
    j = day_idx - k
    out = np.full(len(day_idx), np.nan)
    ok = j >= 0
    out[ok] = mat[j[ok], slot[ok]]
    return out


def build_features(demand: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = demand.merge(weather, on="ts", how="left").sort_values("ts").reset_index(drop=True)
    df = df.rename(columns={"temperature_2m": "temp", "shortwave_radiation": "rad",
                            "wind_speed_10m": "wind", "cloud_cover": "cloud"})
    dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    day_idx = dates.get_indexer(df["date"]).astype(int)
    slot = df["slot"].to_numpy()
    nd = _wide(df, "ND", dates)

    # --- demand-derived features: every one looks back >= 2 days
    df["lag_d2"] = _lookup(nd, day_idx, slot, 2)
    df["lag_d7"] = _lookup(nd, day_idx, slot, 7)
    df["lag_d14"] = _lookup(nd, day_idx, slot, 14)
    df["lag_wd4_mean"] = np.mean(np.vstack([_lookup(nd, day_idx, slot, k) for k in (7, 14, 21, 28)]), axis=0)
    dmean, dmax = np.nanmean(nd, axis=1), np.nanmax(nd, axis=1)
    wk = pd.Series(dmean).rolling(7, min_periods=7).mean().to_numpy()  # mean of days t-6..t
    back2 = np.clip(day_idx - 2, 0, None)
    df["d2_daymean"], df["d2_daymax"], df["wk_mean_d2"] = dmean[back2], dmax[back2], wk[back2]
    early = day_idx < 28
    df.loc[early, ["lag_wd4_mean", "wk_mean_d2", "d2_daymean", "d2_daymax"]] = np.nan
    cap = df.groupby("date")["solar_cap"].last().reindex(dates).ffill()
    df["solar_cap_d2"] = cap.to_numpy()[back2]

    # --- weather (forecast series; all of D is known at 09:00 D-1)
    df["hdd"] = (HDD_BASE - df["temp"]).clip(lower=0)
    df["cdd"] = (df["temp"] - CDD_BASE).clip(lower=0)
    df["solar_pot"] = df["rad"] * df["solar_cap_d2"] / 1e6  # radiation x installed capacity
    g = df.groupby("date")["temp"]
    dm = pd.DataFrame({"m": g.mean(), "lo": g.min(), "hi": g.max()}).reindex(dates)
    df["temp_dmean"], df["temp_dmin"], df["temp_dmax"] = (dm[c].to_numpy()[day_idx] for c in ("m", "lo", "hi"))
    df["temp_prev_dmean"] = dm["m"].to_numpy()[np.clip(day_idx - 1, 0, None)]
    df["temp_ma3"] = dm["m"].rolling(3, min_periods=1).mean().to_numpy()[day_idx]

    # --- calendar
    d = df["date"]
    df["dow"], df["month"] = d.dt.dayofweek, d.dt.month
    doy = d.dt.dayofyear
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    hol = set(HOLIDAYS)
    df["holiday"] = d.isin(hol).astype(int)
    df["hol_eve"] = (d + pd.Timedelta(days=1)).isin(hol).astype(int)
    df["hol_after"] = (d - pd.Timedelta(days=1)).isin(hol).astype(int)
    diff = (d - pd.to_datetime(d.dt.year.astype(str) + "-12-25")).dt.days
    diff = diff.where(diff.abs() <= 183, diff - np.sign(diff) * 365)  # signed days to nearest Christmas
    df["xmas_days"] = diff.clip(-14, 14)
    df["xmas_window"] = (((d.dt.month == 12) & (d.dt.day >= 24)) | ((d.dt.month == 1) & (d.dt.day <= 2))).astype(int)
    df["trend"] = (d - pd.Timestamp("2022-01-01")).dt.days / 365.25
    df["day_n"] = df.groupby("date")["ND"].transform("size")
    return df


def assert_no_leakage(demand: pd.DataFrame, weather: pd.DataFrame, day: str = "2025-03-12") -> None:
    """Corrupt every actual on D-1 and D; features of day D must be unchanged."""
    d = pd.Timestamp(day)
    sub = demand[demand["date"].between(d - pd.Timedelta(days=60), d)].copy()
    a = build_features(sub, weather)
    sub2 = sub.copy()
    m = sub2["date"] >= d - pd.Timedelta(days=1)
    for c in ("ND", "TSD", "solar_gen", "wind_gen"):
        sub2.loc[m, c] = sub2.loc[m, c] * 3 + 12345
    b = build_features(sub2, weather)
    fa, fb = a[a["date"] == d][FEATURES], b[b["date"] == d][FEATURES]
    assert np.allclose(fa.to_numpy(float), fb.to_numpy(float), equal_nan=True), "LEAKAGE detected"


def ridge_design(X: pd.DataFrame) -> pd.DataFrame:
    """Calendar + weather design matrix with slot interactions (for the linear model)."""
    n = len(X)
    S = np.zeros((n, 48))
    S[np.arange(n), X["slot"].to_numpy().astype(int)] = 1
    off = ((X["dow"] >= 5) | (X["holiday"] == 1)).to_numpy().astype(float)
    sat, sun = (X["dow"] == 5).to_numpy().astype(float), (X["dow"] == 6).to_numpy().astype(float)
    blocks = {"slot": S, "slot_off": S * off[:, None], "slot_sat": S * sat[:, None],
              "slot_sun": S * sun[:, None], "slot_hdd": S * X["hdd"].to_numpy()[:, None],
              "slot_cdd": S * X["cdd"].to_numpy()[:, None],
              "slot_solar": S * X["solar_pot"].to_numpy()[:, None],
              "slot_wind": S * X["wind"].to_numpy()[:, None]}
    cols = {f"{name}_{j}": arr[:, j] for name, arr in blocks.items() for j in range(48)}
    M = pd.DataFrame(cols, index=X.index)
    for k in range(7):
        M[f"dow_{k}"] = (X["dow"] == k).astype(float)
    for k in range(1, 13):
        M[f"month_{k}"] = (X["month"] == k).astype(float)
    for c in ["holiday", "hol_eve", "hol_after", "xmas_days", "xmas_window", "trend", "doy_sin", "doy_cos",
              "temp_dmean", "temp_dmin", "temp_dmax", "temp_prev_dmean", "temp_ma3", "cloud", "rad",
              "solar_cap_d2"] + LAGS:
        M[c] = X[c].to_numpy()
    return M
