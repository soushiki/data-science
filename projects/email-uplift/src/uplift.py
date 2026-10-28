"""Uplift learners (S / T / X) and Qini evaluation over repeated stratified 70/30 splits."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from src.config import (ARMS, CLEAN, CURVES, FEATURES, N_SPLITS, N_SPLITS_SPEND, SEED, TEST_SIZE)
from src.utils import Enc, save_metrics

LEARNERS = ["S-learner", "T-learner", "X-learner"]
GRID = np.linspace(0, 1, 101)        # fractions targeted (coefficient integration)
SHOW = np.arange(0, 101, 2)          # 51 points kept for plotting
HGB = dict(max_iter=120, learning_rate=0.05, max_depth=3, min_samples_leaf=100, l2_regularization=1.0)
ARM_KEY = {1: "mens", 2: "womens"}


def make_splits(df: pd.DataFrame, n: int):
    """Stratified (arm x visit) 70/30 splits; seed s is shared by uplift and policy stages."""
    strat = (df["arm_code"] * 2 + df["visit"]).values
    idx = np.arange(len(df))
    for s in range(n):
        yield s, *train_test_split(idx, test_size=TEST_SIZE, stratify=strat, random_state=SEED + s)


def _model(binary: bool, seed: int):
    cls = HistGradientBoostingClassifier if binary else HistGradientBoostingRegressor
    return cls(random_state=seed, **HGB)


def _pred(m, X, binary: bool) -> np.ndarray:
    return m.predict_proba(X)[:, 1] if binary else m.predict(X)


def tau_hat(learner: str, Xtr: np.ndarray, ttr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
            binary: bool, seed: int = 0) -> np.ndarray:
    """Predicted treatment effect for rows of Xte (treatment = 1 vs control = 0)."""
    if learner == "S-learner":
        m = _model(binary, seed).fit(np.column_stack([Xtr, ttr]), ytr)
        n = len(Xte)
        return (_pred(m, np.column_stack([Xte, np.ones(n)]), binary)
                - _pred(m, np.column_stack([Xte, np.zeros(n)]), binary))
    t1, t0 = ttr == 1, ttr == 0
    m1 = _model(binary, seed).fit(Xtr[t1], ytr[t1])
    m0 = _model(binary, seed).fit(Xtr[t0], ytr[t0])
    if learner == "T-learner":
        return _pred(m1, Xte, binary) - _pred(m0, Xte, binary)
    if learner == "X-learner":
        d1 = ytr[t1] - _pred(m0, Xtr[t1], binary)
        d0 = _pred(m1, Xtr[t0], binary) - ytr[t0]
        tau1 = _model(False, seed).fit(Xtr[t1], d1)
        tau0 = _model(False, seed).fit(Xtr[t0], d0)
        g = ttr.mean()    # propensity (constant by design: randomized)
        return g * tau0.predict(Xte) + (1 - g) * tau1.predict(Xte)
    raise ValueError(learner)


def qini(tau: np.ndarray, t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Qini curve Q(f) = incremental outcome per customer in the whole holdout when the top f
    fraction (by predicted uplift) is treated; coefficient = area between Q and the random line."""
    n = len(tau)
    order = np.argsort(-tau, kind="stable")
    ts, ys = t[order], y[order]
    cT, cC = np.cumsum(ts), np.cumsum(1 - ts)
    yT, yC = np.cumsum(ys * ts), np.cumsum(ys * (1 - ts))
    k = np.clip(np.ceil(GRID * n).astype(int) - 1, 0, n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(GRID == 0, 0.0,
                     (np.where(cT[k] > 0, yT[k] / cT[k], 0) - np.where(cC[k] > 0, yC[k] / cC[k], 0))
                     * (k + 1) / n)
    q[0] = 0.0
    coef = float(np.trapezoid(q - GRID * q[-1], GRID))
    auuc = float(np.trapezoid(q, GRID))
    return q, coef, auuc


def _one_split(df: pd.DataFrame, s: int, tr: np.ndarray, te: np.ndarray, outcome: str) -> list[dict]:
    binary = outcome == "visit"
    X = df[FEATURES].values.astype(float)
    arm, y = df["arm_code"].values, df[outcome].values.astype(float)
    rng = np.random.default_rng(SEED + 1000 + s)
    rows = []
    for a in (1, 2):
        mtr = tr[np.isin(arm[tr], (0, a))]
        mte = te[np.isin(arm[te], (0, a))]
        ttr, tte = (arm[mtr] == a).astype(float), (arm[mte] == a).astype(float)
        scores = {L: tau_hat(L, X[mtr], ttr, y[mtr], X[mte], binary, seed=SEED + s) for L in LEARNERS}
        scores["Random"] = rng.random(len(mte))
        for L, tau in scores.items():
            q, coef, auuc = qini(tau, tte, y[mte])
            rows.append({"split": s, "arm": ARM_KEY[a], "learner": L, "qini_coef": coef, "auuc": auuc,
                         "q": q[SHOW].tolist(), "q_total": float(q[-1])})
    return rows


def run(df: pd.DataFrame, outcome: str, n_splits: int) -> list[dict]:
    res = Parallel(n_jobs=-1)(delayed(_one_split)(df, s, tr, te, outcome)
                              for s, tr, te in make_splits(df, n_splits))
    return [r for rs in res for r in rs]


def summarise(rows: list[dict]) -> tuple[list[dict], dict]:
    d = pd.DataFrame(rows)
    table, curves = [], {}
    for (arm, L), g in d.groupby(["arm", "learner"]):
        qa = np.array(g["q"].tolist())
        table.append({"arm": arm, "learner": L, "qini_mean": g["qini_coef"].mean(),
                      "qini_sd": g["qini_coef"].std(ddof=1), "auuc_mean": g["auuc"].mean(),
                      "auuc_sd": g["auuc"].std(ddof=1), "pct_splits_positive": float((g["qini_coef"] > 0).mean()),
                      "n_splits": len(g)})
        curves.setdefault(arm, {})[L] = {"mean": qa.mean(0).tolist(), "p10": np.percentile(qa, 10, axis=0).tolist(),
                                         "p90": np.percentile(qa, 90, axis=0).tolist(),
                                         "random_line_mean": (np.array(g["q_total"])[:, None]
                                                              * GRID[SHOW][None, :]).mean(0).tolist()}
    return table, curves


def best_by_arm(table: list[dict]) -> dict:
    t = pd.DataFrame([r for r in table if r["learner"] != "Random"])
    return {arm: g.loc[g["qini_mean"].idxmax(), "learner"] for arm, g in t.groupby("arm")}


def main() -> None:
    df = pd.read_parquet(CLEAN)
    out, curves = {}, {"grid": GRID[SHOW].tolist()}
    for outcome, n in (("visit", N_SPLITS), ("spend", N_SPLITS_SPEND)):
        rows = run(df, outcome, n)
        table, cv = summarise(rows)
        best = best_by_arm(table)
        t = pd.DataFrame([r for r in table if r["learner"] != "Random"])
        overall = t.groupby("learner")["qini_mean"].mean()
        out[outcome] = {"n_splits": n, "table": table, "best_by_arm": best,
                        "best_overall": overall.idxmax(), "mean_qini_by_learner": overall.to_dict()}
        curves[outcome] = cv
        print(outcome, pd.DataFrame(table).round(5).to_string())
    out["primary_outcome"] = "visit"
    out["best_learner"] = out["visit"]["best_overall"]
    out["features"] = FEATURES
    out["hgb_params"] = HGB
    out["note"] = ("Qini coefficient = area between the Qini curve and the random-targeting line "
                   "(units: incremental outcome per customer; visit = incremental visit probability). "
                   "'Random' = shuffled scores; its mean~0 and sd show the noise floor. "
                   "Best learner is selected on these same holdouts (3 candidates), so its mean is mildly optimistic.")
    save_metrics("uplift", out)
    CURVES.write_text(json.dumps(curves, cls=Enc))
    print("best learner (visit):", out["best_learner"], out["visit"]["best_by_arm"])


if __name__ == "__main__":
    main()
