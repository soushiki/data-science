"""Feature engineering for filing-time risk prediction.

LEAKAGE AVOIDANCE: every feature is computable at the instant a request is filed.
We NEVER use this request's `closed_date`, `status`, `resolution_description`,
`resolution_action_updated_date`, `resolution_hours` or `is_open`.
The one feature that touches closure data -- agency backlog -- only counts *earlier*
requests of the same agency that were still open at this request's `created_date`
(closed_date > created_date of the current row), which is what a live system knows.
Target-encoding statistics (zip) and top-N category lists are learned from the
training period only (out-of-fold for the training rows themselves).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.model_selection import KFold

CAT_COLS = ["agency", "complaint_type", "descriptor", "location_type", "borough",
            "open_data_channel_type"]
TOP_N = {"complaint_type": 40, "descriptor": 60, "location_type": 25}
NUM_COLS = ["hour", "weekday", "is_weekend", "is_holiday", "latitude", "longitude",
            "agency_backlog_7d", "agency_volume_7d", "agency_backlog_share", "zip_te"]
WINDOW = pd.Timedelta(days=7)


def _backlog_one_agency(created: np.ndarray, closed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For time-sorted requests of ONE agency return (open_in_prev_7d, volume_in_prev_7d).

    open_i = #{j: t_i-7d <= created_j < t_i and (closed_j is NaT or closed_j > t_i)}.
    Sweep in time; a Fenwick tree over created-rank counts the window requests that
    have already closed by t_i, so open = window_size - closed_in_window.
    """
    n = len(created)
    lo = np.searchsorted(created, created - WINDOW.to_timedelta64(), side="left")
    hi = np.searchsorted(created, created, side="left")  # strictly earlier
    vol = hi - lo
    # closure events sorted by time
    cl_idx = np.flatnonzero(~np.isnat(closed))
    cl_idx = cl_idx[np.argsort(closed[cl_idx], kind="stable")]
    cl_time = closed[cl_idx]
    tree = [0] * (n + 1)

    def add(i: int) -> None:
        i += 1
        while i <= n:
            tree[i] += 1
            i += i & -i

    def pref(i: int) -> int:  # sum of positions [0, i)
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s

    ptr, out = 0, np.zeros(n, dtype=np.int64)
    for i in range(n):
        t = created[i]
        while ptr < len(cl_idx) and cl_time[ptr] <= t:
            add(int(cl_idx[ptr]))
            ptr += 1
        closed_in_win = pref(int(hi[i])) - pref(int(lo[i]))
        out[i] = vol[i] - closed_in_win
    return out, vol


def agency_backlog(df: pd.DataFrame) -> pd.DataFrame:
    """Backlog columns aligned to df.index (agency-wise, no cross-agency info)."""
    res = pd.DataFrame(index=df.index, columns=["agency_backlog_7d", "agency_volume_7d"],
                       dtype="int64")
    for _, g in df.groupby("agency", sort=False):
        g = g.sort_values("created_date", kind="stable")
        o, v = _backlog_one_agency(g["created_date"].to_numpy("datetime64[ns]"),
                                   g["closed_date"].to_numpy("datetime64[ns]"))
        res.loc[g.index, "agency_backlog_7d"] = o
        res.loc[g.index, "agency_volume_7d"] = v
    return res.astype("int64")


def backlog_surge(df: pd.DataFrame, backlog: pd.Series) -> pd.Series:
    """Backlog relative to the agency's trailing 28-day mean backlog (past only)."""
    out = pd.Series(1.0, index=df.index)
    for _, g in df.groupby("agency", sort=False):
        g = g.sort_values("created_date", kind="stable")
        b = pd.Series(backlog.loc[g.index].to_numpy(float), index=g["created_date"].to_numpy())
        m = b.rolling("28D", closed="left").mean().to_numpy()
        out.loc[g.index] = (b.to_numpy() / (m + 1.0))
    return out.fillna(1.0)


def base_features(df: pd.DataFrame, surge: bool = False) -> pd.DataFrame:
    """Stateless filing-time features (no fitting needed)."""
    out = df[CAT_COLS + ["latitude", "longitude"]].copy()
    ts = df["created_date"]
    out["hour"] = ts.dt.hour
    out["weekday"] = ts.dt.weekday
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    hol = USFederalHolidayCalendar().holidays("2026-01-01", "2026-12-31")
    out["is_holiday"] = ts.dt.normalize().isin(hol).astype(int)
    bl = agency_backlog(df[["agency", "created_date", "closed_date"]])
    out = out.join(bl)
    out["agency_backlog_share"] = out["agency_backlog_7d"] / out["agency_volume_7d"].clip(lower=1)
    out["incident_zip"] = df["incident_zip"]
    if surge:
        out["agency_backlog_surge"] = backlog_surge(df, out["agency_backlog_7d"])
    return out


class FeatureBuilder:
    """Takes the output of base_features computed on the FULL timeline (so backlog
    at the start of a split sees earlier months), then learns train-only top-N category lists and zip target encoding."""

    def __init__(self, smoothing: float = 50.0, seed: int = 42, surge: bool = False) -> None:
        self.smoothing, self.seed, self.surge = smoothing, seed, surge

    @property
    def num_cols(self) -> list[str]:
        return NUM_COLS + (["agency_backlog_surge"] if self.surge else [])

    def _zip_stats(self, zips: pd.Series, y: pd.Series) -> pd.DataFrame:
        g = pd.DataFrame({"z": zips, "y": y}).groupby("z")["y"].agg(["sum", "count"])
        return g

    def _encode(self, zips: pd.Series, stats: pd.DataFrame, prior: float) -> pd.Series:
        s = zips.map(stats["sum"]).fillna(0.0)
        c = zips.map(stats["count"]).fillna(0.0)
        return ((s + self.smoothing * prior) / (c + self.smoothing)).astype(float)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        X = X.copy()
        self.top_: dict[str, list[str]] = {
            c: X[c].value_counts().head(n).index.tolist() for c, n in TOP_N.items()}
        self.prior_ = float(y.mean())
        self.zip_stats_ = self._zip_stats(X["incident_zip"], y)
        te = pd.Series(np.nan, index=X.index)
        for tr, va in KFold(5, shuffle=True, random_state=self.seed).split(X):
            st = self._zip_stats(X["incident_zip"].iloc[tr], y.iloc[tr])
            te.iloc[va] = self._encode(X["incident_zip"].iloc[va], st, self.prior_).to_numpy()
        X["zip_te"] = te
        return self._finish(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["zip_te"] = self._encode(X["incident_zip"], self.zip_stats_, self.prior_)
        return self._finish(X)

    def _finish(self, X: pd.DataFrame) -> pd.DataFrame:
        for c, keep in self.top_.items():
            X[c] = X[c].where(X[c].isin(keep), "Other")
        for c in CAT_COLS:
            X[c] = X[c].astype(str)
        return X[CAT_COLS + self.num_cols]
