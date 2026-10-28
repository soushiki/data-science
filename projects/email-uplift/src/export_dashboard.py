"""Export a compact dashboard.json (< 300 KB) for the static site."""
from __future__ import annotations

import json

from src.config import CURVES, DASHBOARD, EMAIL_COST, GROSS_MARGIN, N_BOOT, N_BOOT_POLICY, QUALITY, SEGMENTS, TEST_SIZE
from src.utils import Enc, load_metrics


def r(x, n=4):
    return None if x is None else round(float(x), n)


def main() -> None:
    m, q = load_metrics(), json.loads(QUALITY.read_text())
    cv = json.loads(CURVES.read_text())
    e, u, p = m["experiment"], m["uplift"], m["policy"]
    tests = e["ab"]["tests"]
    by = {(t["arm"], t["outcome"]): t for t in tests}
    pol = p["primary_split"]["policies"]
    agg = p["across_splits"]["policies"]
    best_pol = max(pol, key=lambda k: pol[k]["incremental_per_1000"])

    kpis = {"n_customers": q["n_rows"], "n_per_arm": e["srm"]["counts"],
            "control_visit_rate": r(e["arm_summary"][0]["visit"]), "control_conv_rate": r(e["arm_summary"][0]["conversion"]),
            "control_spend": r(e["arm_summary"][0]["spend"]),
            "mens_visit_lift_rel": r(by["mens", "visit"]["lift_rel"]), "womens_visit_lift_rel": r(by["womens", "visit"]["lift_rel"]),
            "mens_spend_lift_abs": r(by["mens", "spend"]["lift_abs"]), "womens_spend_lift_abs": r(by["womens", "spend"]["lift_abs"]),
            "srm_p": r(e["srm"]["p_value"]), "max_abs_smd": r(e["balance"]["max_abs_smd"]),
            "best_learner": u["best_learner"], "best_policy_primary_split": best_pol,
            "model_policy_vs_mens_all_ci": [r(p["primary_split"]["model_vs"]["Mens to all"]["ci_low"], 1),
                                            r(p["primary_split"]["model_vs"]["Mens to all"]["ci_high"], 1)]}

    lifts = []
    for t in tests + e["ab"]["head_to_head"]:
        lifts.append({"arm": t["arm"], "vs": t["vs"], "outcome": t["outcome"], "mean_treated": r(t["mean_treated"], 5),
                      "mean_control": r(t["mean_control"], 5), "lift_abs": r(t["lift_abs"], 5), "lift_rel": r(t["lift_rel"]),
                      "ci": [r(t["ci_low"], 5), r(t["ci_high"], 5)], "p": t["p_value"], "p_holm": t["p_holm"],
                      "boot_ci": [r(t["boot_ci_low"], 5), r(t["boot_ci_high"], 5)],
                      "boot_rel_ci": [r(t["boot_rel_ci_low"]), r(t["boot_rel_ci_high"])],
                      "adj_lift_abs": r(t["adj_lift_abs"], 5), "adj_ci": [r(t["adj_ci_low"], 5), r(t["adj_ci_high"], 5)],
                      "ci_width_reduction_pct": r(t["ci_width_reduction_pct"], 2), "covariate_r2": r(t["covariate_r2"], 4)})

    power = [{k: (r(v, 5) if isinstance(v, float) else v) for k, v in row.items()} for row in e["power"]["rows"]]
    seg = [{"arm": c["arm"], "outcome": c["outcome"], "segment": c["segment"], "level": c["level"],
            "n": c["n_treated"] + c["n_control"], "cate": r(c["lift_abs"], 5), "ci": [r(c["ci_low"], 5), r(c["ci_high"], 5)],
            "p": r(c["p_value"], 5)} for c in e["heterogeneity"]["cells"]]
    inter = [{k: (r(v, 5) if isinstance(v, float) else v) for k, v in i.items()} for i in e["heterogeneity"]["interactions"]]

    curves = {"grid": [r(g, 2) for g in cv["grid"]]}
    for oc in ("visit", "spend"):
        curves[oc] = {arm: {L: {k: [r(v, 6) for v in vals] for k, vals in c.items()} for L, c in d.items()}
                      for arm, d in cv[oc].items()}
    learners = {oc: [{k: (r(v, 5) if isinstance(v, float) else v) for k, v in row.items()} for row in u[oc]["table"]]
                for oc in ("visit", "spend")}

    policy = [{"policy": k, "incremental_per_1000": r(v["incremental_per_1000"], 1), "ci": [r(v["ci_low"], 1), r(v["ci_high"], 1)],
               "mean_over_splits": r(agg[k]["mean_incremental_per_1000"], 1), "sd_over_splits": r(agg[k]["sd_across_splits"], 1),
               "share_emailed": r(agg[k]["share_emailed"], 3), "share_mens": r(agg[k]["share_mens"], 3),
               "share_womens": r(agg[k]["share_womens"], 3)} for k, v in pol.items()]
    model_vs = {k: {kk: r(vv, 3) for kk, vv in v.items()} for k, v in p["primary_split"]["model_vs"].items()}

    out = {"kpis": kpis, "arm_summary": [{k: (r(v, 5) if isinstance(v, float) else v) for k, v in a.items()} for a in e["arm_summary"]],
           "srm": e["srm"], "balance": {**{k: v for k, v in e["balance"].items() if k != "rows"},
                                        "rows": [{**b, "smd": r(b["smd"], 4), "mean_treated": r(b["mean_treated"], 4),
                                                  "mean_control": r(b["mean_control"], 4)} for b in e["balance"]["rows"]]},
           "lifts": lifts, "power": power, "heterogeneity": {"cells": seg, "interactions": inter,
                                                             "note": e["heterogeneity"]["note"],
                                                             "n_nominally_significant": e["heterogeneity"]["n_nominally_significant"],
                                                             "n_cells": e["heterogeneity"]["n_cells"],
                                                             "segment_names": SEGMENTS},
           "qini_curves": curves, "learners": learners,
           "best_by_arm": {"visit": u["visit"]["best_by_arm"], "spend": u["spend"]["best_by_arm"]},
           "policy": {"rows": policy, "model_vs": model_vs, "learner": p["learner"],
                      "chosen_k": p["across_splits"]["chosen_k"], "chosen_source": p["across_splits"]["chosen_source"],
                      "rule_split0": p["primary_split"]["rule"], "note": p["method_note"]},
           "assumptions": {"email_cost_per_send": EMAIL_COST, "gross_margin": GROSS_MARGIN,
                           "holdout_share": TEST_SIZE, "lift_bootstrap_resamples": N_BOOT,
                           "policy_bootstrap_resamples": N_BOOT_POLICY, "assignment_probability_per_arm": "1/3",
                           "note": "Cost and margin are illustrative assumptions, not part of the dataset. "
                                   "Outcomes cover 2 weeks after the send."},
           "data_quality": {"n_rows": q["n_rows"], "duplicate_rows": q["full_row_duplicates"],
                            "checks": q["checks"], "spend_zero_share": r(q["spend_zero_share"])}}
    DASHBOARD.write_text(json.dumps(out, separators=(",", ":"), cls=Enc))
    kb = DASHBOARD.stat().st_size / 1024
    print(f"dashboard.json {kb:.0f} KB")
    assert kb < 300, "dashboard.json too large"


if __name__ == "__main__":
    main()
