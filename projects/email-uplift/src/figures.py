"""Publication-quality figures (white bg, gray + single accent, 150 dpi) -- same style as project 1."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import CURVES, FIGURES, SEGMENTS
from src.utils import load_metrics

ACCENT, GRAY, LIGHT, BLUE = "#d9480f", "#6b7280", "#d1d5db", "#1d4ed8"
ARM_COL = {"mens": BLUE, "womens": ACCENT}
ARM_LABEL = {"mens": "Mens e-mail", "womens": "Womens e-mail"}
COV_LABEL = {"recency": "Recency (months)", "history": "Past-year spend ($)", "mens": "Bought men's line",
             "womens": "Bought women's line", "newbie": "New customer", "zip_suburban": "Zip: Suburban",
             "zip_urban": "Zip: Urban", "chan_phone": "Channel: Phone", "chan_web": "Channel: Web"}
OUT_LABEL = {"visit": "Site visit", "conversion": "Conversion", "spend": "Spend"}


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


def fig_balance(m: dict) -> None:
    rows = m["experiment"]["balance"]["rows"]
    covs = list(COV_LABEL)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvspan(-0.1, 0.1, color="#f3f4f6", zorder=0)
    ax.axvline(0, color=GRAY, lw=1)
    for off, arm in ((-0.12, "mens"), (0.12, "womens")):
        r = {x["covariate"]: x["smd"] for x in rows if x["arm"] == arm}
        ax.scatter([r[c] for c in covs], np.arange(len(covs)) + off, color=ARM_COL[arm], s=34,
                   label=f"{ARM_LABEL[arm]} vs control", zorder=3)
    ax.set_yticks(range(len(covs)), [COV_LABEL[c] for c in covs])
    ax.invert_yaxis()
    ax.set_xlim(-0.15, 0.15)
    ax.set_xlabel("Standardized mean difference (shaded band = |SMD| < 0.1)")
    ax.set_title("Randomization worked: every covariate is balanced")
    ax.legend(loc="lower right")
    _save(fig, "balance_smd.png")


def fig_lift_forest(m: dict) -> None:
    tests = m["experiment"]["ab"]["tests"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ylab = []
    for i, t in enumerate(tests):
        y = len(tests) - 1 - i
        c = ARM_COL[t["arm"]]
        mc = t["mean_control"]
        ax.plot([100 * t["ci_low"] / mc, 100 * t["ci_high"] / mc], [y + 0.14] * 2, color=LIGHT, lw=5,
                solid_capstyle="butt", label="Raw 95% CI" if i == 0 else None)
        ax.plot([100 * t["adj_ci_low"] / mc, 100 * t["adj_ci_high"] / mc], [y - 0.14] * 2, color=c, lw=2.2,
                label="Covariate-adjusted 95% CI" if i == 0 else None)
        ax.scatter([100 * t["lift_abs"] / mc], [y + 0.14], color=GRAY, s=22, zorder=3)
        ax.scatter([100 * t["adj_lift_abs"] / mc], [y - 0.14], color=c, s=22, zorder=3)
        sig = "Holm p<0.001" if t["p_holm"] < 0.001 else f"Holm p={t['p_holm']:.3f}"
        ax.text(1.01, y, sig, transform=ax.get_yaxis_transform(), va="center", fontsize=9, color=GRAY)
        ylab.append(f"{ARM_LABEL[t['arm']]}: {OUT_LABEL[t['outcome']].lower()}")
    ax.set_yticks(range(len(tests)), ylab[::-1])
    ax.axvline(0, color="#111827", ls=":", lw=1)
    ax.set_xlabel("Relative lift vs. control (%)")
    ax.set_title("Both e-mails lift every outcome; adjustment barely tightens CIs")
    ax.legend(loc="upper right")
    _save(fig, "lift_forest.png")


def fig_segment_cate(m: dict) -> None:
    cells = m["experiment"]["heterogeneity"]["cells"]
    levels = [(s, l) for s in SEGMENTS for l in sorted({c["level"] for c in cells if c["segment"] == s})]
    fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharey=True)
    for ax, (y, unit, scale) in zip(axes, (("visit", "pp", 100), ("spend", "$", 1))):
        for j, (s, l) in enumerate(levels):
            for off, arm in ((-0.17, "mens"), (0.17, "womens")):
                c = next(x for x in cells if x["arm"] == arm and x["outcome"] == y
                         and x["segment"] == s and x["level"] == l)
                ax.plot([scale * c["ci_low"], scale * c["ci_high"]], [j + off] * 2, color=ARM_COL[arm], lw=1.6)
                ax.scatter([scale * c["lift_abs"]], [j + off], color=ARM_COL[arm], s=16, zorder=3,
                           label=ARM_LABEL[arm] if j == 0 else None)
        ax.axvline(0, color="#111827", ls=":", lw=1)
        ax.set_xlabel("Visit uplift (percentage points)" if y == "visit" else "Spend uplift ($ per customer)")
        ax.set_title("Visit CATE" if y == "visit" else "Spend CATE")
        prev = None
        for j, (s, l) in enumerate(levels):
            if prev is not None and s != prev:
                ax.axhline(j - 0.5, color=LIGHT, lw=0.8)
            prev = s
    axes[0].set_yticks(range(len(levels)), [f"{SEGMENTS[s]}: {l}".replace("$", r"\$") for s, l in levels], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper right")
    fig.suptitle("Segment-level effects (95% CI, not multiplicity-adjusted)", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    _save(fig, "segment_cate_forest.png")


def fig_qini(m: dict) -> None:
    cv = json.loads(CURVES.read_text())
    grid = np.array(cv["grid"])
    best = m["uplift"]["visit"]["best_by_arm"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    for ax, arm in zip(axes, ("mens", "womens")):
        L = best[arm]
        c = cv["visit"][arm][L]
        ax.fill_between(100 * grid, 100 * np.array(c["p10"]), 100 * np.array(c["p90"]), color=ARM_COL[arm],
                        alpha=0.18, label="p10-p90 across 20 splits")
        ax.plot(100 * grid, 100 * np.array(c["mean"]), color=ARM_COL[arm], lw=2.4, label=f"{L} (mean)")
        ax.plot(100 * grid, 100 * np.array(c["random_line_mean"]), color=GRAY, ls="--", lw=1.4,
                label="Random targeting")
        row = next(r for r in m["uplift"]["visit"]["table"] if r["arm"] == arm and r["learner"] == L)
        ax.set_title(f"{ARM_LABEL[arm]}: Qini {row['qini_mean'] * 100:.2f} +/- {row['qini_sd'] * 100:.2f} pp")
        ax.set_xlabel("% of customers e-mailed (highest predicted uplift first)")
        ax.set_ylabel("Incremental visits per 100 customers")
        ax.legend(loc="upper left", fontsize=9)
    _save(fig, "qini_curves.png")


def fig_policy(m: dict) -> None:
    p = m["policy"]["primary_split"]["policies"]
    ag = m["policy"]["across_splits"]["policies"]
    names = list(p)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(names))[::-1]
    for yi, n in zip(y, names):
        v = p[n]
        col = ACCENT if n.startswith("Model") else GRAY
        ax.barh(yi, v["incremental_per_1000"], color=col, alpha=0.85, height=0.55)
        ax.plot([v["ci_low"], v["ci_high"]], [yi, yi], color="#111827", lw=1.4)
        ax.scatter([ag[n]["mean_incremental_per_1000"]], [yi], marker="D", color="white", edgecolor="#111827",
                   s=34, zorder=4, label="Mean over 10 random splits" if yi == y[0] else None)
    ax.set_yticks(y, names)
    ax.axvline(0, color="#111827", lw=1)
    ax.set_xlabel("Incremental profit per 1,000 customers vs. e-mailing nobody ($)")
    ax.set_title("Policy value: differences sit inside the noise")
    ax.legend(loc="lower right")
    _save(fig, "policy_value.png")


def fig_power(m: dict) -> None:
    rows = m["experiment"]["power"]["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, o in zip(axes, ("visit", "conversion", "spend")):
        rs = [r for r in rows if r["outcome"] == o]
        x = np.arange(len(rs))
        ax.bar(x - 0.2, [100 * r["mde_rel"] for r in rs], 0.38, color=LIGHT, label="MDE (80% power)")
        ax.bar(x + 0.2, [100 * r["observed_lift_rel"] for r in rs], 0.38,
               color=[ARM_COL[r["arm"]] for r in rs], label="Observed lift")
        ax.set_xticks(x, [ARM_LABEL[r["arm"]].replace(" e-mail", "") for r in rs])
        ax.set_title(OUT_LABEL[o])
        ax.set_ylabel("Relative to control (%)" if o == "visit" else "")
        if o == "visit":
            ax.legend(loc="upper right", fontsize=9)
        for xi, r in zip(x, rs):
            ax.text(xi + 0.2, 100 * r["observed_lift_rel"], f"{r['observed_lift_rel']:.0%}", ha="center",
                    va="bottom", fontsize=9)
    fig.suptitle("Minimum detectable effect vs. observed lift (n ~ 21k per arm)", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    _save(fig, "power_mde.png")


def main() -> None:
    style()
    m = load_metrics()
    for f in (fig_balance, fig_lift_forest, fig_segment_cate, fig_qini, fig_policy, fig_power):
        f(m)
    print("figures:", sorted(p.name for p in FIGURES.glob("*.png")))


if __name__ == "__main__":
    main()
