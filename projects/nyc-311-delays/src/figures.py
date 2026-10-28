"""Publication-quality figures (white bg, gray + single accent, 150 dpi)."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from src.config import CLEAN, FIGURES, METRICS, PREDS, PREDS2
from src.features import agency_backlog

ACCENT, GRAY, LIGHT = "#d9480f", "#6b7280", "#d1d5db"
MODEL_COLORS = {"baseline_complaint_rate": LIGHT, "logistic_regression": GRAY,
                "hist_gradient_boosting": ACCENT}
MODEL_LABELS = {"baseline_complaint_rate": "Baseline (type rate)",
                "logistic_regression": "Logistic regression",
                "hist_gradient_boosting": "Gradient boosting"}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRAY,
        "axes.labelcolor": "#111827", "xtick.color": "#374151", "ytick.color": "#374151",
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 150,
        "savefig.dpi": 150, "axes.grid": False})


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)


def pct(ax: plt.Axes, axis: str = "x") -> None:
    fmt = mtick.PercentFormatter(1.0, decimals=0)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def fig_complaint_types(df: pd.DataFrame) -> None:
    g = df.groupby("complaint_type")["missed_72h"].agg(["mean", "size"])
    g = g.sort_values("size", ascending=False).head(15).sort_values("mean")
    base = df["missed_72h"].mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(g.index, g["mean"], color=[ACCENT if v > base else GRAY for v in g["mean"]])
    ax.axvline(base, color="#111827", ls="--", lw=1)
    ax.text(base, len(g) - 0.4, f" overall {base:.0%}", fontsize=10, va="bottom")
    pct(ax)
    ax.set_xlabel("Share of requests unresolved after 72 hours")
    ax.set_title("Missed-72h rate, 15 highest-volume complaint types")
    _save(fig, "missed_rate_by_complaint_type.png")


def fig_heatmap(df: pd.DataFrame) -> None:
    piv = (df.assign(h=df["created_date"].dt.hour, d=df["created_date"].dt.day_of_week)
             .pivot_table(index="d", columns="h", values="missed_72h", aggfunc="mean"))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(piv.values, aspect="auto", cmap="Oranges")
    ax.set_yticks(range(7), DAYS)
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_xlabel("Hour request was filed")
    ax.set_title("Missed-72h rate by filing time")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    cb.outline.set_visible(False)
    _save(fig, "missed_rate_hour_weekday.png")


def fig_borough_hours(df: pd.DataFrame) -> None:
    g = (df[df["borough"] != "Unknown"].groupby("borough")["resolution_hours"]
         .median().sort_values())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(g.index, g.values, color=[ACCENT if v == g.max() else GRAY for v in g.values])
    for i, v in enumerate(g.values):
        ax.text(v, i, f" {v:.1f} h", va="center", fontsize=11)
    ax.set_xlabel("Median resolution time (hours, closed requests)")
    ax.set_title("Median time to resolution by borough")
    _save(fig, "median_resolution_by_borough.png")


def fig_pr(pred: pd.DataFrame, metrics: dict, name: str = "pr_curves.png",
           title: str = "Precision-recall on held-out August") -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODEL_COLORS:
        p, r, _ = precision_recall_curve(pred["y"], pred[m])
        ap = metrics["test"][m]["pr_auc"]
        ax.plot(r, p, color=MODEL_COLORS[m], lw=2.2 if m.startswith("hist") else 1.8,
                label=f"{MODEL_LABELS[m]} (AP {ap:.2f})")
    ax.axhline(pred["y"].mean(), color="#111827", ls=":", lw=1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.set_ylim(0, 1.02)
    _save(fig, name)


def fig_calibration(pred: pd.DataFrame, metrics: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color="#111827", ls=":", lw=1, label="Perfect")
    for m in ("logistic_regression", "hist_gradient_boosting"):
        fy, fx = calibration_curve(pred["y"], pred[m], n_bins=15, strategy="quantile")
        ax.plot(fx, fy, "o-", color=MODEL_COLORS[m], ms=4, lw=2,
                label=f"{MODEL_LABELS[m]} (Brier {metrics['test'][m]['brier']:.3f})")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed missed rate")
    ax.set_title("Calibration on held-out August")
    ax.legend(loc="upper left")
    _save(fig, "calibration.png")


def fig_importance(metrics: dict, name: str = "permutation_importance.png") -> None:
    imp = metrics["permutation_importance"][:12][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh([d["feature"] for d in imp], [d["importance"] for d in imp],
            xerr=[d["std"] for d in imp], color=ACCENT, ecolor=GRAY)
    ax.set_xlabel("Drop in PR-AUC when feature is shuffled")
    ax.set_title("Permutation importance (test set)")
    _save(fig, name)


def fig_lift(metrics: dict) -> None:
    """Cumulative gains: share of delayed requests captured vs share flagged."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], color="#111827", ls=":", lw=1, label="Random")
    for m, pts in metrics["gains_curves"].items():
        a = np.array([[0, 0]] + pts)
        ax.plot(a[:, 0], a[:, 1], color=MODEL_COLORS[m], lw=2.4 if m.startswith("hist") else 1.8,
                label=MODEL_LABELS[m])
    lift = metrics["test"]["hist_gradient_boosting"]["lift_top10"]
    ax.axvline(0.10, color=LIGHT, lw=1)
    ax.text(0.11, 0.05, f"Top 10% flagged: gradient boosting lift {lift:.2f}x", fontsize=10)
    pct(ax)
    pct(ax, "y")
    ax.set_xlabel("Share of requests flagged (highest risk first)")
    ax.set_ylabel("Share of delayed requests captured")
    ax.set_title("Cumulative gains, delayed vs peers")
    ax.legend(loc="lower right")
    _save(fig, "study2_lift.png")


def fig_backlog(df: pd.DataFrame) -> None:
    d = df[["agency", "created_date", "closed_date", "missed_72h"]].copy()
    d["backlog"] = agency_backlog(d)["agency_backlog_7d"]
    # within-agency deciles so agencies of different scale are comparable
    d["bin"] = d.groupby("agency")["backlog"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) if len(s) >= 100 else np.nan)
    g = d.dropna(subset=["bin"]).groupby("bin")["missed_72h"].mean()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot((g.index + 1).astype(int), g.values, "o-", color=ACCENT, lw=2.2)
    pct(ax, "y")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Agency backlog at filing time (decile within agency; 10 = busiest)")
    ax.set_ylabel("Missed-72h rate")
    ax.set_title("Agency backlog vs missed-72h rate")
    _save(fig, "backlog_vs_missed.png")


def main() -> None:
    style()
    df = pd.read_parquet(CLEAN)
    pred = pd.read_parquet(PREDS)
    allm = json.loads(METRICS.read_text())
    metrics = allm["study1_72h"]
    fig_complaint_types(df)
    fig_heatmap(df)
    fig_borough_hours(df)
    fig_pr(pred, metrics)
    fig_calibration(pred, metrics)
    fig_importance(metrics)
    fig_backlog(df)
    m2, pred2 = allm["study2_delay_vs_peers"], pd.read_parquet(PREDS2)
    fig_pr(pred2, m2, "study2_pr_curves.png", "Precision-recall, delayed vs peers (August)")
    fig_importance(m2, "study2_importance.png")
    fig_lift(m2)
    print("figures saved to", FIGURES)


if __name__ == "__main__":
    main()
