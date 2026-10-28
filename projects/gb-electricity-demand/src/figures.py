"""Publication-quality figures (white bg, gray + single accent, 150 dpi) -- same style as project 1."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from src.config import FIGURES, METRICS, PREDS
from src.train import FEATURES_PQ

ACCENT, GRAY, LIGHT = "#d9480f", "#6b7280", "#d1d5db"
SEASONS = {"Winter (Dec-Feb)": ([12, 1, 2], "#1d4ed8"), "Spring (Mar-May)": ([3, 4, 5], "#6b7280"),
           "Summer (Jun-Aug)": ([6, 7, 8], ACCENT), "Autumn (Sep-Nov)": ([9, 10, 11], "#a16207")}
LABELS = {"naive": "Seasonal naive", "ridge": "Ridge", "hgb": "Gradient boosting"}


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


def gw(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))


def hour_axis(ax: plt.Axes) -> None:
    ax.set_xticks(range(0, 48, 6), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_xlim(0, 47)


def fig_temp_scatter(df: pd.DataFrame) -> None:
    d = df.groupby("date").agg(nd=("ND", "mean"), t=("temp", "mean"), dow=("dow", "first"),
                               hol=("holiday", "first"), n=("ND", "size"))
    d = d[d["n"] == 48]
    wd = (d["dow"] < 5) & (d["hol"] == 0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(d.loc[~wd, "t"], d.loc[~wd, "nd"], s=9, color=LIGHT, label="Weekends & bank holidays")
    ax.scatter(d.loc[wd, "t"], d.loc[wd, "nd"], s=9, color=ACCENT, alpha=0.6, label="Working days")
    ax.axvline(15.5, color="#111827", ls=":", lw=1)
    ax.text(15.7, d["nd"].max(), "15.5 C heating base", fontsize=9, va="top")
    gw(ax)
    ax.set_xlabel("Population-weighted forecast temperature (C, daily mean)")
    ax.set_ylabel("Average national demand (GW)")
    ax.set_title("Demand rises when it is cold; the summer side stays flat")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.9))
    _save(fig, "demand_vs_temperature.png")


def fig_profiles(df: pd.DataFrame) -> None:
    wd = df[(df["dow"] < 5) & (df["holiday"] == 0)]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for name, (months, col) in SEASONS.items():
        g = wd[wd["month"].isin(months)].groupby("slot")["ND"].mean()
        ax.plot(g.index, g.values, color=col, lw=2.2, label=name)
    hour_axis(ax)
    gw(ax)
    ax.set_xlabel("Time of day (local clock)")
    ax.set_ylabel("Average national demand (GW)")
    ax.set_title("Average working-day demand profile by season")
    ax.legend(loc="upper left", ncol=2)
    _save(fig, "daily_profile_by_season.png")


def fig_duck(df: pd.DataFrame) -> None:
    s = df[(df["month"].isin([6, 7, 8])) & (df["dow"] < 5) & (df["holiday"] == 0)].copy()
    s["year"] = s["date"].dt.year
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    yrs = sorted(s["year"].unique())
    cols = np.linspace(0.25, 1, len(yrs))
    for y, c in zip(yrs, cols):
        g = s[s["year"] == y].groupby("slot")["ND"].mean()
        ax.plot(g.index, g.values, color=plt.cm.Oranges(c), lw=2.2, label=str(y))
    hour_axis(ax)
    gw(ax)
    ax.set_ylabel("Average national demand (GW)")
    ax.set_xlabel("Time of day (local clock)")
    ax.set_title("Summer working days: the midday belly deepens")
    ax.legend(title=None, loc="upper left", ncol=2)
    ax = axes[1]
    mid = s[(s["slot"] >= 24) & (s["slot"] < 28)].groupby("year")["ND"].mean()
    eve = s[(s["slot"] >= 36) & (s["slot"] < 40)].groupby("year")["ND"].mean()
    cap = s.groupby("year")["solar_cap"].mean() / 1000
    x = np.arange(len(yrs))
    ax.bar(x - 0.2, mid.values, 0.4, color=ACCENT, label="Midday (12:00-14:00)")
    ax.bar(x + 0.2, eve.values, 0.4, color=LIGHT, label="Evening (18:00-20:00)")
    ax.set_xticks(x, [str(y) for y in yrs])
    ax.set_ylim(mid.min() * 0.8, eve.max() * 1.04)
    gw(ax)
    ax.set_title("Midday vs evening demand")
    ax.legend(loc="upper right", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(x, cap.values, "o--", color="#111827", lw=1.2, ms=4)
    ax2.set_ylabel("Embedded solar capacity (GW, dashed)")
    ax2.spines["right"].set_visible(True)
    ax2.set_ylim(0, cap.max() * 1.15)
    _save(fig, "solar_duck.png")


def fig_forecast_window(pred: pd.DataFrame, start: str = "2026-01-12", days: int = 14) -> None:
    s = pred[(pred["date"] >= start) & (pred["date"] < pd.Timestamp(start) + pd.Timedelta(days=days))]
    t = s["ts"].dt.tz_convert("Europe/London").dt.tz_localize(None)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.fill_between(t, s["p10c"], s["p90c"], color=ACCENT, alpha=0.18, lw=0, label="80% interval")
    ax.plot(t, s["y"], color="#111827", lw=1.4, label="Actual")
    ax.plot(t, s["hgb"], color=ACCENT, lw=1.4, label="Model forecast (made 09:00 on D-1)")
    gw(ax)
    ax.set_ylabel("National demand (GW)")
    ax.set_title(f"Two-week test window, {s['date'].min():%d %b} - {s['date'].max():%d %b %Y}")
    ax.legend(loc="lower left", ncol=3)
    ax.margins(x=0)
    _save(fig, "forecast_window.png")


def fig_peak_h2h(metrics: dict) -> None:
    p = metrics["peak"]
    mo = pd.DataFrame(p["by_month"])
    x = np.arange(len(mo))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(x - 0.2, mo["neso_mae"], 0.4, color=GRAY, label=f"NESO official (overall {p['neso']['mae']:,.0f} MW)")
    ax.bar(x + 0.2, mo["model_mae"], 0.4, color=ACCENT, label=f"Our model (overall {p['hgb']['mae']:,.0f} MW)")
    ax.set_xticks(x, [pd.Timestamp(m + "-01").strftime("%b\n%y") for m in mo["month"]])
    ax.set_ylabel("Daily-peak MAE (MW)")
    ax.set_title("Daily peak error by month: model vs NESO day-ahead forecast")
    ax.legend(loc="upper right")
    _save(fig, "peak_error_vs_neso.png")


def fig_importance(metrics: dict, top: int = 14) -> None:
    imp = pd.Series(metrics["importance"]).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(imp.index, imp.values, color=[ACCENT if i >= len(imp) - 3 else GRAY for i in range(len(imp))])
    ax.set_xlabel("Increase in MAE when shuffled (MW)")
    ax.set_title("Permutation importance (test set)")
    _save(fig, "feature_importance.png")


def fig_residual_hour(pred: pd.DataFrame) -> None:
    r = pred.assign(e=pred["hgb"] - pred["y"])
    g = r.groupby("slot")["e"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(g.mean().index, g.quantile(0.1), g.quantile(0.9), color=LIGHT, alpha=0.7, lw=0,
                    label="10th-90th percentile")
    ax.plot(g.mean().index, g.mean().values, color=ACCENT, lw=2.2, label="Mean error")
    ax.axhline(0, color="#111827", lw=1)
    hour_axis(ax)
    ax.set_xlabel("Time of day (local clock)")
    ax.set_ylabel("Forecast - actual (MW)")
    ax.set_title("Forecast error by time of day (test set)")
    ax.legend(loc="lower left", ncol=2)
    _save(fig, "residual_by_hour.png")


def fig_models(metrics: dict) -> None:
    t = metrics["test"]
    ks = list(LABELS)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.barh([LABELS[k] for k in ks], [t[k]["mae"] for k in ks], color=[LIGHT, GRAY, ACCENT])
    for i, k in enumerate(ks):
        ax.text(t[k]["mae"], i, f" {t[k]['mae']:,.0f} MW ({t[k]['mape']:.1%})", va="center", fontsize=10)
    ax.set_xlim(0, max(t[k]["mae"] for k in ks) * 1.3)
    ax.set_xlabel("Half-hourly MAE on test set (MW)")
    ax.set_title("Model comparison")
    _save(fig, "model_comparison.png")


def main() -> None:
    style()
    df = pd.read_parquet(FEATURES_PQ)
    pred = pd.read_parquet(PREDS)
    metrics = json.loads(METRICS.read_text())
    fig_temp_scatter(df)
    fig_profiles(df)
    fig_duck(df)
    fig_forecast_window(pred)
    fig_peak_h2h(metrics)
    fig_importance(metrics)
    fig_residual_hour(pred)
    fig_models(metrics)
    print("figures saved to", FIGURES)


if __name__ == "__main__":
    main()
