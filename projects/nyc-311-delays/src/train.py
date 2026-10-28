"""Time-based split, three models, evaluation, permutation importance, for two studies.

Study 1: missed_72h (resolution > 72h or still open).
Study 2: delayed_vs_peers (slower than the train p75 for the same complaint type).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import CLEAN, METRICS, MODEL, MODEL2, PREDS, PREDS2, QUALITY, SEED, THRESHOLDS2
from src.features import CAT_COLS, FeatureBuilder, base_features

TOP_FRAC = 0.10


def split_by_time(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Train = Mar-Jun, validation = Jul, test = Aug (no shuffling)."""
    m = df["created_date"].dt.month
    return {"train": df[m.between(3, 6)], "val": df[m == 7], "test": df[m == 8]}


def top_frac_metrics(y: np.ndarray, score: np.ndarray, frac: float = TOP_FRAC) -> dict[str, float]:
    """Precision / recall / lift when flagging the top `frac` highest-scored."""
    k = int(round(len(y) * frac))
    idx = np.argsort(-score, kind="stable")[:k]
    tp = y[idx].sum()
    prec = tp / k
    return {"precision_top10": float(prec), "recall_top10": float(tp / y.sum()),
            "lift_top10": float(prec / y.mean())}


def evaluate(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {"roc_auc": float(roc_auc_score(y, score)),
            "pr_auc": float(average_precision_score(y, score)),
            "brier": float(brier_score_loss(y, np.clip(score, 0, 1))),
            **top_frac_metrics(y, score)}


class RateBaseline:
    """Historical positive-rate per complaint_type (raw type, train only)."""

    def __init__(self, target: str = "missed_72h") -> None:
        self.target = target

    def fit(self, df: pd.DataFrame) -> "RateBaseline":
        self.rates_ = df.groupby("complaint_type")[self.target].mean()
        self.prior_ = float(df[self.target].mean())
        return self

    def predict_proba1(self, df: pd.DataFrame) -> np.ndarray:
        return df["complaint_type"].map(self.rates_).fillna(self.prior_).to_numpy(float)


def make_logreg(num_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CAT_COLS),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num_cols)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.5, max_iter=500))])


def make_hgb(num_cols: list[str], **kw) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                encoded_missing_value=-1), CAT_COLS)],
        remainder="passthrough")
    flags = [True] * len(CAT_COLS) + [False] * len(num_cols)
    clf = HistGradientBoostingClassifier(categorical_features=flags, random_state=SEED,
                                         early_stopping=True, validation_fraction=0.1,
                                         max_iter=400, **kw)
    return Pipeline([("pre", pre), ("clf", clf)])


def tune_hgb(num_cols, Xtr, ytr, Xva, yva) -> tuple[Pipeline, dict]:
    """Small grid, selected on validation PR-AUC."""
    best, best_ap, best_p = None, -1.0, {}
    for lr, leaves, l2 in itertools.product([0.05, 0.1], [31, 63], [0.0, 1.0]):
        p = {"learning_rate": lr, "max_leaf_nodes": leaves, "l2_regularization": l2}
        m = make_hgb(num_cols, **p).fit(Xtr, ytr)
        ap = average_precision_score(yva, m.predict_proba(Xva)[:, 1])
        print(f"  hgb {p} val PR-AUC={ap:.4f}")
        if ap > best_ap:
            best, best_ap, best_p = m, ap, p
    return best, best_p


def gains_curve(y: np.ndarray, score: np.ndarray, n: int = 50) -> list[list[float]]:
    """(fraction flagged, fraction of all positives captured) at n points."""
    order = np.argsort(-score, kind="stable")
    cum = np.cumsum(y[order]) / y.sum()
    idx = np.unique(np.linspace(0, len(y) - 1, n).astype(int))
    return [[round((i + 1) / len(y), 4), round(float(cum[i]), 4)] for i in idx]


def run_study(df: pd.DataFrame, target: str, surge: bool, model_path: Path,
              preds_path: Path) -> dict:
    """Fit baseline / logreg / HGB on train, tune on val, evaluate on test."""
    df = df.reset_index(drop=True)
    parts = split_by_time(df)
    sizes = {k: len(v) for k, v in parts.items()}
    print("split sizes", sizes)
    B = base_features(df, surge)  # full timeline so backlog sees earlier months
    fb = FeatureBuilder(seed=SEED, surge=surge)
    num_cols = fb.num_cols
    X = {"train": fb.fit_transform(B.loc[parts["train"].index], parts["train"][target])}
    for k in ("val", "test"):
        X[k] = fb.transform(B.loc[parts[k].index])
    y = {k: parts[k][target].to_numpy() for k in parts}

    base = RateBaseline(target).fit(parts["train"])
    lr = make_logreg(num_cols).fit(X["train"], y["train"])
    print("tuning HGB")
    hgb, hgb_params = tune_hgb(num_cols, X["train"], y["train"], X["val"], y["val"])

    scorers = {
        "baseline_complaint_rate": lambda k: base.predict_proba1(parts[k]),
        "logistic_regression": lambda k: lr.predict_proba(X[k])[:, 1],
        "hist_gradient_boosting": lambda k: hgb.predict_proba(X[k])[:, 1]}
    metrics: dict = {"target": target, "split_sizes": sizes, "hgb_params": hgb_params,
                     "base_rate": {k: float(y[k].mean()) for k in y}, "val": {}, "test": {}}
    for name, fn in scorers.items():
        metrics["val"][name] = evaluate(y["val"], fn("val"))
        metrics["test"][name] = evaluate(y["test"], fn("test"))
        print(name, {k: round(v, 4) for k, v in metrics["test"][name].items()})

    # never deploy the naive baseline: pick between the two real models on val PR-AUC
    cand = ["logistic_regression", "hist_gradient_boosting"]
    best_name = max(cand, key=lambda n: metrics["val"][n]["pr_auc"])
    metrics["best_model"] = best_name
    best = {"logistic_regression": lr, "hist_gradient_boosting": hgb}[best_name]

    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(X["test"]), size=min(30_000, len(X["test"])), replace=False)
    pi = permutation_importance(best, X["test"].iloc[idx], y["test"][idx], n_repeats=3,
                                scoring="average_precision", random_state=SEED, n_jobs=1)
    imp = sorted(zip(X["test"].columns, pi.importances_mean, pi.importances_std),
                 key=lambda t: -t[1])
    metrics["permutation_importance"] = [
        {"feature": f, "importance": float(m), "std": float(s)} for f, m, s in imp]
    print("top features", [f for f, _, _ in imp[:5]])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"name": best_name, "pipeline": best, "feature_builder": fb,
                 "target": target}, model_path)
    pred = parts["test"][["unique_key", "created_date", "complaint_type", "borough",
                          "latitude", "longitude", target]].rename(columns={target: "y"})
    pred = pred.reset_index(drop=True)
    for name, fn in scorers.items():
        pred[name] = fn("test")
    pred.to_parquet(preds_path, index=False)
    metrics["gains_curves"] = {n: gains_curve(y["test"], pred[n].to_numpy()) for n in scorers}
    return metrics


def study2_target(df: pd.DataFrame, q: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop 0s closures; label requests slower than the train p75 of their complaint type.

    Thresholds use closed requests in the training months only; types with <200 train
    rows share one pooled 'Other' threshold. Still-open requests count as delayed.
    """
    n = len(df)
    df = df[~df["closed_instant"]].copy()
    q["study2_dropped_closed_instant"] = n - len(df)
    train = df[df["created_date"].dt.month.between(3, 6)]
    counts = train["complaint_type"].value_counts()
    small = set(counts[counts < 200].index)
    df["_grp"] = df["complaint_type"].where(~df["complaint_type"].isin(small), "Other")
    tr = df.loc[train.index]
    thr = tr.dropna(subset=["resolution_hours"]).groupby("_grp")["resolution_hours"].quantile(0.75)
    df["p75_threshold_h"] = df["_grp"].map(thr).fillna(thr.get("Other", thr.median()))
    df["delayed_vs_peers"] = (df["is_open"] | (df["resolution_hours"] > df["p75_threshold_h"]).fillna(False)).astype(int)
    q["study2_n_pooled_types"] = len(small)
    q["study2_base_rate"] = float(df["delayed_vs_peers"].mean())
    thr_df = thr.rename("p75_hours").reset_index().rename(columns={"_grp": "complaint_type"})
    return df, thr_df


def main() -> None:
    df = pd.read_parquet(CLEAN)
    print("== Study 1: missed 72h ==")
    s1 = run_study(df, "missed_72h", False, MODEL, PREDS)
    print("== Study 2: delayed vs peers ==")
    q = json.loads(QUALITY.read_text())
    df2, thr = study2_target(df, q)
    thr.to_parquet(THRESHOLDS2, index=False)
    s2 = run_study(df2, "delayed_vs_peers", True, MODEL2, PREDS2)
    QUALITY.write_text(json.dumps(q, indent=2))
    METRICS.write_text(json.dumps({"study1_72h": s1, "study2_delay_vs_peers": s2}, indent=2))


if __name__ == "__main__":
    main()
