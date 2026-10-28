"""Train, validate and test day-ahead half-hourly demand models; head-to-head vs NESO.

Splits (by target date): train 2022-01 -> 2024-12, validation 2025-01 -> 2025-06,
test 2025-07 -> latest. Hyper-parameters are tuned on validation only; final models are
refit on train+val and scored once on test. Features obey the D-1 09:00 leakage rules
documented in `src.features`.
"""
from __future__ import annotations

import itertools
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (CLEAN, METRICS, MODEL, NESO_CLEAN, PREDS, ROOT, SEED, TEST_START,
                        TRAIN_END, VAL_END, VAL_START, WEATHER_CLEAN)
from src.features import FEATURES, assert_no_leakage, build_features, ridge_design

FEATURES_PQ = ROOT / "data" / "processed" / "features.parquet"
NAMES = {"naive": "Seasonal naive (t-7d)", "ridge": "Ridge (calendar+weather)",
         "hgb": "Gradient boosting"}


def load() -> pd.DataFrame:
    dem, w = pd.read_parquet(CLEAN), pd.read_parquet(WEATHER_CLEAN)
    assert_no_leakage(dem, w)
    print("leakage check passed")
    df = build_features(dem, w)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    df.to_parquet(FEATURES_PQ, index=False)
    return df


def m(y, p) -> dict:
    e = np.asarray(p) - np.asarray(y)
    return {"mae": float(np.abs(e).mean()), "mape": float((np.abs(e) / np.asarray(y)).mean()),
            "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean())}


def hgb(**kw) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=SEED, min_samples_leaf=40, l2_regularization=1.0, **kw)


def day_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-day actual and predicted peak (max) and trough (min) of half-hourly series."""
    g = df.groupby("date")
    out = pd.DataFrame({"act_peak": g["y"].max(), "act_min": g["y"].min()})
    for c in cols:
        out[f"{c}_peak"], out[f"{c}_min"] = g[c].max(), g[c].min()
    return out


def peak_report(tab: pd.DataFrame, val_tab: pd.DataFrame, holidays: set) -> dict:
    """Head-to-head daily peak (and trough) vs the NESO day-ahead forecast."""
    t = tab.dropna(subset=["neso_peak"]).copy()
    v = val_tab.dropna(subset=["neso_peak"])
    bias_neso = float((v["neso_peak"] - v["act_peak"]).mean())   # estimated on validation only
    bias_model = float((v["hgb_peak"] - v["act_peak"]).mean())
    t["neso_bc_peak"] = t["neso_peak"] - bias_neso
    t["hgb_bc_peak"] = t["hgb_peak"] - bias_model
    t["blend_peak"] = (t["hgb_peak"] + t["neso_peak"]) / 2
    ys = t["act_peak"]
    rep: dict = {"n_days": int(len(t)), "val_bias_neso": bias_neso, "val_bias_model": bias_model,
                 "first_day": str(t.index.min().date()), "last_day": str(t.index.max().date())}
    for c in ("naive", "ridge", "hgb", "hgb_bc", "neso", "neso_bc", "blend"):
        rep[c] = m(ys, t[f"{c}_peak"])
    for c in ("hgb", "neso"):
        e = np.abs(t[f"{c}_peak"] - ys)
        rep[f"{c}_p90_abs_err"] = float(e.quantile(0.9))
    ae_h, ae_n = (t["hgb_peak"] - ys).abs(), (t["neso_peak"] - ys).abs()
    rep["pct_days_model_beats_neso"] = float((ae_h < ae_n).mean())
    rep["pct_days_model_beats_neso_bc"] = float(((t["hgb_peak"] - ys).abs() < (t["neso_bc_peak"] - ys).abs()).mean())
    rep["pct_days_model_bc_beats_neso_bc"] = float(((t["hgb_bc_peak"] - ys).abs() < (t["neso_bc_peak"] - ys).abs()).mean())
    dp = t.dropna(subset=["neso_dp"])
    rep["dp_only"] = {"n_days": int(len(dp)), "neso_dp": m(dp["act_peak"], dp["neso_dp"]),
                      "hgb": m(dp["act_peak"], dp["hgb_peak"]),
                      "pct_days_model_beats": float(((dp["hgb_peak"] - dp["act_peak"]).abs()
                                                     < (dp["neso_dp"] - dp["act_peak"]).abs()).mean())}
    # by month / day type
    t["month"] = t.index.strftime("%Y-%m")
    t["daytype"] = np.where(t.index.isin(holidays), "Bank holiday",
                            np.where(t.index.day_of_week == 6, "Sunday",
                                     np.where(t.index.day_of_week == 5, "Saturday", "Weekday")))
    for key in ("month", "daytype"):
        rows = []
        for k, g in t.groupby(key):
            rows.append({key: k, "n": int(len(g)),
                         "model_mae": float((g["hgb_peak"] - g["act_peak"]).abs().mean()),
                         "neso_mae": float((g["neso_peak"] - g["act_peak"]).abs().mean()),
                         "model_bias": float((g["hgb_peak"] - g["act_peak"]).mean()),
                         "neso_bias": float((g["neso_peak"] - g["act_peak"]).mean()),
                         "model_win": float(((g["hgb_peak"] - g["act_peak"]).abs()
                                             < (g["neso_peak"] - g["act_peak"]).abs()).mean())})
        rep[f"by_{key}"] = rows
    # overnight trough
    tm = tab.dropna(subset=["neso_min"])
    rep["trough"] = {"n_days": int(len(tm)), "model": m(tm["act_min"], tm["hgb_min"]),
                     "neso": m(tm["act_min"], tm["neso_min"]),
                     "pct_days_model_beats_neso": float(((tm["hgb_min"] - tm["act_min"]).abs()
                                                          < (tm["neso_min"] - tm["act_min"]).abs()).mean())}
    return rep


def main() -> None:
    from src.features import HOLIDAYS
    df = load()
    d = df["date"]
    tr = df[d <= TRAIN_END]
    va = df[(d >= VAL_START) & (d <= VAL_END)]
    te = df[d >= TEST_START]
    trva = df[d <= VAL_END]
    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,}  ({te['date'].min().date()} -> {te['date'].max().date()})")
    y = "ND"
    res: dict = {"splits": {
        "train": [str(tr["date"].min().date()), TRAIN_END, int(len(tr))],
        "val": [VAL_START, VAL_END, int(len(va))],
        "test": [str(te["date"].min().date()), str(te["date"].max().date()), int(len(te))]}}

    # ---- ridge: tune alpha on validation
    Rtr, Rva = ridge_design(tr[FEATURES]), ridge_design(va[FEATURES])
    best_r = min(((a, m(va[y], make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(Rtr, tr[y]).predict(Rva))["mae"])
                  for a in (1, 10, 100, 1000, 3000)), key=lambda t: t[1])
    alpha = best_r[0]
    print("ridge alpha", alpha, "val MAE", round(best_r[1]))

    # ---- HGB: light grid on validation
    grid = list(itertools.product((0.05, 0.1), (31, 63), (300, 600)))
    scores = []
    for lr, leaves, it in grid:
        mdl = hgb(learning_rate=lr, max_leaf_nodes=leaves, max_iter=it).fit(tr[FEATURES], tr[y])
        scores.append((m(va[y], mdl.predict(va[FEATURES]))["mae"], lr, leaves, it))
        print(f"  hgb lr={lr} leaves={leaves} iter={it}: val MAE {scores[-1][0]:.0f}")
    _, lr, leaves, it = min(scores)
    hp = dict(learning_rate=lr, max_leaf_nodes=leaves, max_iter=it)
    res["hyperparameters"] = {"ridge_alpha": alpha, "hgb": hp}

    # ---- validation predictions from train-only models (also used for bias / CQR calibration)
    ridge_tr = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Rtr, tr[y])
    hgb_tr = hgb(**hp).fit(tr[FEATURES], tr[y])
    qkw = dict(loss="quantile", **hp)
    lo_tr = hgb(quantile=0.1, **qkw).fit(tr[FEATURES], tr[y])
    hi_tr = hgb(quantile=0.9, **qkw).fit(tr[FEATURES], tr[y])
    val = va[["ts", "date", "slot", "period", y]].rename(columns={y: "y"}).copy()
    val["naive"], val["ridge"] = va["lag_d7"].to_numpy(), ridge_tr.predict(Rva)
    val["hgb"] = hgb_tr.predict(va[FEATURES])
    val["lo"], val["hi"] = lo_tr.predict(va[FEATURES]), hi_tr.predict(va[FEATURES])
    res["val"] = {k: m(val["y"], val[k]) for k in ("naive", "ridge", "hgb")}
    # conformalised quantile regression offset (80% target) from validation
    score = np.maximum(val["lo"] - val["y"], val["y"] - val["hi"])
    n = len(score)
    cqr = float(np.quantile(score, min(1.0, 0.8 * (1 + 1 / n))))
    res["interval"] = {"cqr_offset_mw": cqr, "val_raw_coverage": float(((val["y"] >= val["lo"]) & (val["y"] <= val["hi"])).mean())}

    # ---- final models: refit on train+val, score on test
    Rtv, Rte = ridge_design(trva[FEATURES]), ridge_design(te[FEATURES])
    ridge_f = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Rtv, trva[y])
    hgb_f = hgb(**hp).fit(trva[FEATURES], trva[y])
    lo_f = hgb(quantile=0.1, **qkw).fit(trva[FEATURES], trva[y])
    hi_f = hgb(quantile=0.9, **qkw).fit(trva[FEATURES], trva[y])
    test = te[["ts", "date", "slot", "period", y]].rename(columns={y: "y"}).copy()
    test["naive"], test["ridge"] = te["lag_d7"].to_numpy(), ridge_f.predict(Rte)
    test["hgb"] = hgb_f.predict(te[FEATURES])
    test["p10"], test["p90"] = lo_f.predict(te[FEATURES]), hi_f.predict(te[FEATURES])
    test["p10c"], test["p90c"] = test["p10"] - cqr, test["p90"] + cqr
    res["test"] = {k: m(test["y"], test[k]) for k in ("naive", "ridge", "hgb")}
    cov = lambda a, b: float(((test["y"] >= test[a]) & (test["y"] <= test[b])).mean())
    res["interval"].update({
        "test_raw_coverage": cov("p10", "p90"), "test_calibrated_coverage": cov("p10c", "p90c"),
        "test_raw_mean_width_mw": float((test["p90"] - test["p10"]).mean()),
        "test_calibrated_mean_width_mw": float((test["p90c"] - test["p10c"]).mean()),
        "test_below_p10_raw": float((test["y"] < test["p10"]).mean()),
        "test_above_p90_raw": float((test["y"] > test["p90"]).mean())})
    test["month"] = test["date"].dt.strftime("%Y-%m")
    res["interval"]["calibrated_coverage_by_month"] = {
        k: float(((g["y"] >= g["p10c"]) & (g["y"] <= g["p90c"])).mean()) for k, g in test.groupby("month")}
    res["test_by_month"] = {k: {mm: m(g["y"], g[mm])["mae"] for mm in ("naive", "ridge", "hgb")}
                            for k, g in test.groupby("month")}

    # ---- head-to-head vs NESO daily peak
    neso = pd.read_parquet(NESO_CLEAN).set_index("date")
    hol = set(HOLIDAYS)
    ttab = day_table(test, ["naive", "ridge", "hgb"]).join(neso[["neso_peak", "neso_min", "neso_dp"]])
    vtab = day_table(val, ["naive", "ridge", "hgb"]).join(neso[["neso_peak", "neso_min", "neso_dp"]])
    res["peak"] = peak_report(ttab, vtab, hol)
    ttab.reset_index().to_parquet(PREDS.with_name("test_daily.parquet"), index=False)

    # ---- permutation importance (final HGB, test subsample, MAE)
    sub = te.sample(6000, random_state=SEED)
    pi = permutation_importance(hgb_f, sub[FEATURES], sub[y], n_repeats=3, random_state=SEED,
                                scoring="neg_mean_absolute_error", n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
    res["importance"] = {k: float(v) for k, v in imp.items()}
    res["best_model"] = min(("ridge", "hgb"), key=lambda k: res["val"][k]["mae"])

    test.drop(columns="month").to_parquet(PREDS, index=False)
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump({"model": hgb_f, "q_lo": lo_f, "q_hi": hi_f, "cqr_offset": cqr, "features": FEATURES,
                 "ridge": ridge_f, "best_model": res["best_model"]}, MODEL)
    METRICS.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("val", "test", "interval")}, indent=1)[:3500])
    p = res["peak"]
    print("PEAK", p["n_days"], "days | model", p["hgb"], "| neso", p["neso"], "| wins", p["pct_days_model_beats_neso"])
    print("top features", list(imp.index[:8]))


if __name__ == "__main__":
    main()
