"""Offline policy evaluation on randomized data: who should get which email?

Policy value is estimated with arm-matching (self-normalised IPW; the randomization probability is
1/3 for every arm, so weights are equal and cancel): mean profit among holdout customers whose
*actual* arm equals the arm the policy would send. The Horvitz-Thompson form (3 x sum(match*profit)/n)
is reported as a sensitivity check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from src.config import (CLEAN, CV_FOLDS, EMAIL_COST, FEATURES, GROSS_MARGIN, K_GRID, N_BOOT_POLICY,
                        N_POLICY_SPLITS, SEED)
from src.uplift import make_splits, tau_hat
from src.utils import load_metrics, save_metrics

POLICIES = ["Email nobody", "Mens to all", "Womens to all", "Rule: gender-history cells",
            "Model-targeted (top-k)"]


def profit(arm: np.ndarray, spend: np.ndarray) -> np.ndarray:
    return GROSS_MARGIN * spend - EMAIL_COST * (arm > 0)


def value_hajek(action: np.ndarray, arm: np.ndarray, p: np.ndarray) -> float:
    m = arm == action
    return float(p[m].mean()) if m.any() else np.nan


def value_ht(action: np.ndarray, arm: np.ndarray, p: np.ndarray) -> float:
    return float(3 * np.sum(p * (arm == action)) / len(arm))


def arm_scores(learner: str, X: np.ndarray, arm: np.ndarray, y: dict, tr: np.ndarray, te: np.ndarray,
               seed: int, spv: float) -> dict:
    """Expected incremental profit per customer of each email arm (rows te), from two sources."""
    out = {}
    for a in (1, 2):
        m = tr[np.isin(arm[tr], (0, a))]
        t = (arm[m] == a).astype(float)
        u_visit = tau_hat(learner, X[m], t, y["visit"][m], X[te], True, seed)
        u_spend = tau_hat(learner, X[m], t, y["spend"][m], X[te], False, seed)
        out[("visit", a)] = GROSS_MARGIN * u_visit * spv - EMAIL_COST
        out[("spend", a)] = GROSS_MARGIN * u_spend - EMAIL_COST
    return out


def decide(score1: np.ndarray, score2: np.ndarray, k_pct: int) -> np.ndarray:
    """Send the arm with the higher predicted profit gain to the top k% of customers by that gain."""
    best = np.maximum(score1, score2)
    arm = np.where(score1 >= score2, 1, 2)
    n_send = int(round(len(best) * k_pct / 100))
    act = np.zeros(len(best), dtype=int)
    if n_send > 0:
        top = np.argsort(-best, kind="stable")[:n_send]
        act[top] = arm[top]
    return act


def rule_policy(df_tr: pd.DataFrame) -> dict:
    """For each (mens, womens) cell choose the action with highest mean train profit."""
    rule = {}
    for (m, w), g in df_tr.groupby(["mens", "womens"]):
        pr = pd.Series(profit(g["arm_code"].values, g["spend"].values), index=g.index)
        means = pr.groupby(g["arm_code"]).mean()
        rule[(int(m), int(w))] = int(means.idxmax())
    return rule


def _split(df: pd.DataFrame, s: int, tr: np.ndarray, te: np.ndarray, learner: str, keep: bool):
    X = df[FEATURES].values.astype(float)
    arm = df["arm_code"].values
    y = {"visit": df["visit"].values.astype(float), "spend": df["spend"].values.astype(float)}
    prof = profit(arm, y["spend"])
    spv = y["spend"][tr].sum() / max(y["visit"][tr].sum(), 1)        # train $ spend per visit

    # --- choose (source, k) on TRAIN by cross-fitting: out-of-fold scores, arm-matching value
    oof = {key: np.zeros(len(tr)) for key in [("visit", 1), ("visit", 2), ("spend", 1), ("spend", 2)]}
    for f, (a, b) in enumerate(KFold(CV_FOLDS, shuffle=True, random_state=SEED + s).split(tr)):
        sc = arm_scores(learner, X, arm, y, tr[a], tr[b], SEED + s + f, spv)
        for key, v in sc.items():
            oof[key][b] = v
    base = value_hajek(np.zeros(len(tr), dtype=int), arm[tr], prof[tr])
    cv = {}
    for src in ("visit", "spend"):
        for k in K_GRID:
            act = decide(oof[(src, 1)], oof[(src, 2)], k)
            cv[(src, k)] = value_hajek(act, arm[tr], prof[tr]) - base
    src_k = max(cv, key=lambda c: (cv[c], -c[1]))
    # --- final scores on holdout
    sc = arm_scores(learner, X, arm, y, tr, te, SEED + s, spv)
    act_model = decide(sc[(src_k[0], 1)], sc[(src_k[0], 2)], src_k[1])
    rule = rule_policy(df.iloc[tr])
    act_rule = np.array([rule[(int(m), int(w))] for m, w in zip(df["mens"].values[te], df["womens"].values[te])])
    n = len(te)
    actions = {"Email nobody": np.zeros(n, dtype=int), "Mens to all": np.ones(n, dtype=int),
               "Womens to all": np.full(n, 2), "Rule: gender-history cells": act_rule,
               "Model-targeted (top-k)": act_model}
    a_te, p_te = arm[te], prof[te]
    v0 = value_hajek(actions["Email nobody"], a_te, p_te)
    res = {"split": s, "chosen_source": src_k[0], "chosen_k": src_k[1], "cv_gain_per_1000": 1000 * cv[src_k],
           "rule": {f"mens={m},womens={w}": ARM[v] for (m, w), v in rule.items()}, "policies": {}}
    for name, act in actions.items():
        res["policies"][name] = {
            "profit_per_1000": 1000 * value_hajek(act, a_te, p_te),
            "incremental_per_1000": 1000 * (value_hajek(act, a_te, p_te) - v0),
            "ht_incremental_per_1000": 1000 * (value_ht(act, a_te, p_te) - value_ht(actions["Email nobody"], a_te, p_te)),
            "share_emailed": float((act > 0).mean()), "share_mens": float((act == 1).mean()),
            "share_womens": float((act == 2).mean()), "n_matched": int((a_te == act).sum())}
    if keep:
        res["_arrays"] = {"arm": a_te, "profit": p_te, "actions": actions}
    return res


ARM = {0: "none", 1: "mens", 2: "womens"}


def bootstrap(arrays: dict, n_boot: int = N_BOOT_POLICY) -> dict:
    arm, p, acts = arrays["arm"], arrays["profit"], arrays["actions"]
    rng = np.random.default_rng(SEED)
    n = len(arm)
    names = list(acts)
    inc = {k: np.empty(n_boot) for k in names}
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        v = {k: value_hajek(acts[k][i], arm[i], p[i]) for k in names}
        for k in names:
            inc[k][b] = 1000 * (v[k] - v["Email nobody"])
    point = {k: 1000 * (value_hajek(acts[k], arm, p) - value_hajek(acts["Email nobody"], arm, p)) for k in names}
    out = {k: {"incremental_per_1000": point[k], "ci_low": float(np.percentile(inc[k], 2.5)),
               "ci_high": float(np.percentile(inc[k], 97.5))} for k in names}
    model = "Model-targeted (top-k)"
    diffs = {}
    for k in names:
        if k == model:
            continue
        d = inc[model] - inc[k]
        diffs[k] = {"diff_per_1000": point[model] - point[k], "ci_low": float(np.percentile(d, 2.5)),
                    "ci_high": float(np.percentile(d, 97.5)), "p_model_not_better": float(np.mean(d <= 0))}
    return {"policies": out, "model_vs": diffs, "n_boot": n_boot, "n_holdout": n}


def main() -> None:
    df = pd.read_parquet(CLEAN)
    learner = load_metrics()["uplift"]["best_learner"]
    res = Parallel(n_jobs=-1)(delayed(_split)(df, s, tr, te, learner, s == 0)
                              for s, tr, te in make_splits(df, N_POLICY_SPLITS))
    primary = bootstrap(res[0].pop("_arrays"))
    for r in res:
        r.pop("_arrays", None)
    agg = {}
    for name in POLICIES:
        inc = np.array([r["policies"][name]["incremental_per_1000"] for r in res])
        ht = np.array([r["policies"][name]["ht_incremental_per_1000"] for r in res])
        agg[name] = {"mean_incremental_per_1000": inc.mean(), "sd_across_splits": inc.std(ddof=1),
                     "mean_ht_incremental_per_1000": ht.mean(),
                     "share_emailed": float(np.mean([r["policies"][name]["share_emailed"] for r in res])),
                     "share_mens": float(np.mean([r["policies"][name]["share_mens"] for r in res])),
                     "share_womens": float(np.mean([r["policies"][name]["share_womens"] for r in res]))}
    model_inc = np.array([r["policies"]["Model-targeted (top-k)"]["incremental_per_1000"] for r in res])
    paired = {}
    for name in POLICIES:
        if name == "Model-targeted (top-k)":
            continue
        d = model_inc - np.array([r["policies"][name]["incremental_per_1000"] for r in res])
        paired[name] = {"mean_diff": d.mean(), "sd": d.std(ddof=1), "n_splits_model_better": int((d > 0).sum())}
    out = {"learner": learner, "assumptions": {"email_cost": EMAIL_COST, "gross_margin": GROSS_MARGIN},
           "primary_split": {"split": 0, **primary, "chosen_source": res[0]["chosen_source"],
                             "chosen_k": res[0]["chosen_k"], "rule": res[0]["rule"]},
           "across_splits": {"n_splits": len(res), "policies": agg, "model_minus_other": paired,
                             "chosen_k": [r["chosen_k"] for r in res],
                             "chosen_source": [r["chosen_source"] for r in res],
                             "rules": [r["rule"] for r in res]},
           "method_note": ("Incremental profit per 1,000 customers vs emailing nobody, on holdout, using "
                           "arm-matching with known 1/3 assignment probability (self-normalised IPW). "
                           "Bootstrap CIs resample holdout customers on split 0 (paired across policies); "
                           "the across-split mean +/- sd shows sensitivity to the random split. "
                           "Costs/margins are illustrative assumptions, not data.")}
    save_metrics("policy", out)
    for k, v in primary["policies"].items():
        print(f"{k:30s} {v['incremental_per_1000']:8.1f}  [{v['ci_low']:.1f}, {v['ci_high']:.1f}]   "
              f"across-splits {agg[k]['mean_incremental_per_1000']:.1f} +/- {agg[k]['sd_across_splits']:.1f}")
    print("chosen", out["across_splits"]["chosen_source"], out["across_splits"]["chosen_k"])
    print("model vs:", primary["model_vs"])


if __name__ == "__main__":
    main()
