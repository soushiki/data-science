"""Randomization checks, A/B readout, variance reduction, power, heterogeneity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.config import ALPHA, ARMS, CLEAN, N_BOOT, POWER, SEED, SEGMENTS
from src.utils import save_metrics

OUTCOMES = ["visit", "conversion", "spend"]
ZA, ZB = stats.norm.ppf(1 - ALPHA / 2), stats.norm.ppf(POWER)
COVS = ["recency", "history", "mens", "womens", "newbie", "zip_suburban", "zip_urban",
        "chan_phone", "chan_web"]
ARM_KEY = {1: "mens", 2: "womens"}


def srm(df: pd.DataFrame) -> dict:
    obs = df["arm_code"].value_counts().sort_index()
    exp = np.full(len(obs), len(df) / len(obs))
    chi, p = stats.chisquare(obs.values, exp)
    return {"counts": {ARMS[k]: int(v) for k, v in obs.items()},
            "shares": {ARMS[k]: float(v / len(df)) for k, v in obs.items()},
            "expected_share": 1 / 3, "chi2": float(chi), "df": len(obs) - 1, "p_value": float(p),
            "srm_flag": bool(p < 0.001)}


def balance(df: pd.DataFrame) -> dict:
    ctl = df[df["arm_code"] == 0]
    rows = []
    for a in (1, 2):
        t = df[df["arm_code"] == a]
        for c in COVS:
            sd = np.sqrt((t[c].var() + ctl[c].var()) / 2)
            smd = (t[c].mean() - ctl[c].mean()) / sd
            rows.append({"arm": ARM_KEY[a], "covariate": c, "mean_treated": float(t[c].mean()),
                         "mean_control": float(ctl[c].mean()), "smd": float(smd),
                         "flag": bool(abs(smd) > 0.1)})
    # omnibus: do covariates jointly predict assignment? (LR test, multinomial logit)
    X = sm.add_constant(df[COVS].astype(float))
    full = sm.MNLogit(df["arm_code"], X).fit(disp=0, maxiter=200)
    null = sm.MNLogit(df["arm_code"], X[["const"]]).fit(disp=0)
    lr = 2 * (full.llf - null.llf)
    dof = len(COVS) * 2
    return {"rows": rows, "max_abs_smd": float(max(abs(r["smd"]) for r in rows)),
            "n_flagged": int(sum(r["flag"] for r in rows)),
            "omnibus_lr_chi2": float(lr), "omnibus_df": dof,
            "omnibus_p_value": float(stats.chi2.sf(lr, dof))}


def _diff_test(yt: np.ndarray, yc: np.ndarray, binary: bool) -> dict:
    nt, nc = len(yt), len(yc)
    mt, mc = yt.mean(), yc.mean()
    d = mt - mc
    if binary:   # two-proportion z (pooled SE for the test, unpooled for the CI)
        pp = (yt.sum() + yc.sum()) / (nt + nc)
        z = d / np.sqrt(pp * (1 - pp) * (1 / nt + 1 / nc))
        p = 2 * stats.norm.sf(abs(z))
        se = np.sqrt(mt * (1 - mt) / nt + mc * (1 - mc) / nc)
        crit = ZA
    else:        # Welch t
        vt, vc = yt.var(ddof=1) / nt, yc.var(ddof=1) / nc
        se = np.sqrt(vt + vc)
        dof = se ** 4 / (vt ** 2 / (nt - 1) + vc ** 2 / (nc - 1))
        p = 2 * stats.t.sf(abs(d / se), dof)
        crit = stats.t.ppf(1 - ALPHA / 2, dof)
    return {"n_treated": nt, "n_control": nc, "mean_treated": mt, "mean_control": mc, "lift_abs": d,
            "lift_rel": d / mc if mc else np.nan, "se": se, "ci_low": d - crit * se,
            "ci_high": d + crit * se, "p_value": p}


def _bootstrap(yt, yc, rng, n_boot=N_BOOT) -> dict:
    nt, nc = len(yt), len(yc)
    diffs, rels = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        mt = yt[rng.integers(0, nt, nt)].mean()
        mc = yc[rng.integers(0, nc, nc)].mean()
        diffs[b], rels[b] = mt - mc, (mt - mc) / mc if mc else np.nan
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    rlo, rhi = np.nanpercentile(rels, [2.5, 97.5])
    return {"boot_ci_low": lo, "boot_ci_high": hi, "boot_rel_ci_low": rlo, "boot_rel_ci_high": rhi,
            "n_boot": n_boot}


def _adjusted(d: pd.DataFrame, treat: np.ndarray, y: str) -> dict:
    """Regression adjustment with pre-period covariates (HC1 robust SE)."""
    X = pd.DataFrame({"const": 1.0, "T": treat, "history": d["history"].values,
                      "log_history": np.log1p(d["history"].values)}, index=d.index)
    for r in range(2, 13):   # recency as dummies (1 = baseline)
        X[f"rec{r}"] = (d["recency"].values == r).astype(float)
    for c in ["mens", "womens", "newbie", "chan_phone", "chan_web"]:
        X[c] = d[c].values.astype(float)
    yv = d[y].values.astype(float)
    res = sm.OLS(yv, X).fit(cov_type="HC1")
    b, se = float(res.params["T"]), float(res.bse["T"])
    r2 = float(sm.OLS(yv, X.drop(columns="T")).fit().rsquared)
    return {"adj_lift_abs": b, "adj_se": se, "adj_ci_low": b - ZA * se, "adj_ci_high": b + ZA * se,
            "adj_p_value": float(res.pvalues["T"]), "covariate_r2": r2}


def compare(df: pd.DataFrame, arm_t: int, arm_c: int, y: str, rng, boot: bool = True) -> dict:
    d = df[df["arm_code"].isin([arm_t, arm_c])]
    t = (d["arm_code"] == arm_t).values
    yv = d[y].values.astype(float)
    out = _diff_test(yv[t], yv[~t], binary=(y != "spend"))
    out["outcome"] = y
    out["var_treated"], out["var_control"] = float(yv[t].var(ddof=1)), float(yv[~t].var(ddof=1))
    if boot:
        out.update(_bootstrap(yv[t], yv[~t], rng))
    out.update(_adjusted(d, t.astype(float), y))
    out["ci_width_raw"] = out["ci_high"] - out["ci_low"]
    out["ci_width_adj"] = out["adj_ci_high"] - out["adj_ci_low"]
    out["ci_width_reduction_pct"] = 100 * (1 - out["ci_width_adj"] / out["ci_width_raw"])
    out["adj_lift_rel"] = out["adj_lift_abs"] / out["mean_control"] if out["mean_control"] else np.nan
    return out


def ab_readout(df: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    tests = []
    for a in (1, 2):
        for y in OUTCOMES:
            r = compare(df, a, 0, y, rng)
            r.update({"arm": ARM_KEY[a], "vs": "control"})
            tests.append(r)
    holm = multipletests([t["p_value"] for t in tests], alpha=ALPHA, method="holm")
    for t, ph, rej in zip(tests, holm[1], holm[0]):
        t["p_holm"], t["holm_reject"] = float(ph), bool(rej)
    h2h = []
    for y in OUTCOMES:
        r = compare(df, 1, 2, y, rng)
        r.update({"arm": "mens", "vs": "womens"})
        h2h.append(r)
    ph = multipletests([t["p_value"] for t in h2h], alpha=ALPHA, method="holm")
    for t, pv, rej in zip(h2h, ph[1], ph[0]):
        t["p_holm"], t["holm_reject"] = float(pv), bool(rej)
    return {"tests": tests, "head_to_head": h2h, "n_tests_holm": len(tests),
            "bootstrap_resamples": N_BOOT, "seed": SEED}


def power(tests: list[dict]) -> dict:
    rows = []
    for t in tests:
        se, d = t["se"], t["lift_abs"]
        mde, mde_adj = (ZA + ZB) * se, (ZA + ZB) * t["adj_se"]
        n_req = (ZA + ZB) ** 2 * (t["var_treated"] + t["var_control"]) / d ** 2 if d != 0 else np.nan
        rows.append({"arm": t["arm"], "outcome": t["outcome"],
                     "n_per_arm_observed": (t["n_treated"] + t["n_control"]) // 2,
                     "mde_abs": mde, "mde_rel": mde / t["mean_control"], "mde_abs_adjusted": mde_adj,
                     "mde_rel_adjusted": mde_adj / t["mean_control"], "observed_lift_abs": d,
                     "observed_lift_rel": t["lift_rel"],
                     "power_at_observed_lift": float(stats.norm.cdf(abs(d) / se - ZA)),
                     "n_per_arm_required_for_observed_lift": n_req})
    return {"alpha": ALPHA, "power": POWER, "rows": rows}


def heterogeneity(df: pd.DataFrame) -> dict:
    cells, inter = [], []
    for a in (1, 2):
        d_arm = df[df["arm_code"].isin([0, a])].copy()
        for y in ("visit", "spend"):
            for seg in SEGMENTS:
                for lvl, g in d_arm.groupby(seg):
                    t = (g["arm_code"] == a).values
                    yv = g[y].values.astype(float)
                    r = _diff_test(yv[t], yv[~t], y == "visit")
                    cells.append({"arm": ARM_KEY[a], "outcome": y, "segment": seg, "level": str(lvl),
                                  **{k: r[k] for k in ("n_treated", "n_control", "mean_treated", "mean_control",
                                                       "lift_abs", "se", "ci_low", "ci_high", "p_value")}})
                m = smf.ols(f"{y} ~ treat * C({seg})", d_arm).fit(cov_type="HC1")
                names = [n for n in m.params.index if ":" in n]
                R = np.zeros((len(names), len(m.params)))
                for i, n in enumerate(names):
                    R[i, list(m.params.index).index(n)] = 1
                w = m.wald_test(R, use_f=True, scalar=True)
                inter.append({"arm": ARM_KEY[a], "outcome": y, "segment": seg, "df": len(names),
                              "p_interaction": float(w.pvalue)})
    ps = [g["p_interaction"] for g in inter]
    for g, pb, ph in zip(inter, multipletests(ps, method="fdr_bh")[1], multipletests(ps, method="holm")[1]):
        g["p_bh"], g["p_holm"] = float(pb), float(ph)
    return {"cells": cells, "interactions": inter, "n_cells": len(cells),
            "n_nominally_significant": int(sum(c["p_value"] < ALPHA for c in cells)),
            "expected_false_positives_at_5pct": 0.05 * len(cells),
            "n_interaction_tests": len(inter),
            "n_interactions_sig_raw": int(sum(i["p_interaction"] < ALPHA for i in inter)),
            "n_interactions_sig_bh": int(sum(i["p_bh"] < ALPHA for i in inter)),
            "n_interactions_sig_holm": int(sum(i["p_holm"] < ALPHA for i in inter)),
            "note": ("Segment levels x 2 arms x 2 outcomes are all pre-specified but cell-level CATEs are "
                     "NOT multiplicity-adjusted; about 5% would look nominally significant by chance. "
                     "Interaction (heterogeneity) tests are BH/Holm-adjusted across the 24 variable-level tests.")}


def arm_summary(df: pd.DataFrame) -> list[dict]:
    g = df.groupby("arm_code").agg(n=("visit", "size"), visit=("visit", "mean"),
                                   conversion=("conversion", "mean"), spend=("spend", "mean"),
                                   spend_sd=("spend", "std"))
    g["spend_per_converter"] = df[df["conversion"] == 1].groupby("arm_code")["spend"].mean()
    return [{"arm": ARMS[k], **{c: float(v) for c, v in r.items()}} for k, r in g.iterrows()]


def main() -> None:
    df = pd.read_parquet(CLEAN)
    out = {"arm_summary": arm_summary(df), "srm": srm(df), "balance": balance(df)}
    out["ab"] = ab_readout(df)
    out["power"] = power(out["ab"]["tests"])
    out["heterogeneity"] = heterogeneity(df)
    save_metrics("experiment", out)
    print("SRM p=%.3f; max|SMD|=%.3f; omnibus p=%.3f" % (out["srm"]["p_value"], out["balance"]["max_abs_smd"],
                                                         out["balance"]["omnibus_p_value"]))
    for t in out["ab"]["tests"]:
        print(f"{t['arm']:7s} {t['outcome']:10s} lift={t['lift_abs']:.4f} rel={t['lift_rel']:.1%} "
              f"p={t['p_value']:.2g} holm={t['p_holm']:.2g} shrink={t['ci_width_reduction_pct']:.1f}%")


if __name__ == "__main__":
    main()
