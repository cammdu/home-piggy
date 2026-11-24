"""Export the calibrated parameters, NL rules and scenario library as JSON for the browser app (web/).

The browser engine (web/engine.js) reads exactly these files. Two things are precomputed here so the
JavaScript port needs no linear algebra: the Cholesky factor of each regime's (PSD-repaired)
correlation matrix, via the same `cholesky_factors` used by the Python engine.

Usage: PYTHONPATH=. python tools/export_web_data.py [--country nl|uk]
Writes web/data/<country>/: params_{repo,hmm}.json, policy_2026.json, scenarios.json, glidepaths.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any

from homepiggy.affordability_nl import load_policy
from homepiggy.affordability_de import FundTax, load_policy_de
from homepiggy.affordability_uk import load_policy_uk
from homepiggy.config import LEVEL_VARS, MarketParams
from homepiggy.market import cholesky_factors
from homepiggy.uncertainty import load_bootstrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "data"
BOOT = {c: {"repo": ROOT / "params" / f"{c}_bootstrap.json", "hmm": ROOT / "params" / f"{c}_bootstrap_hmm.json"}
        for c in ("nl", "uk", "de")}
INF = 1e18     # JSON has no infinity; larger than any price


def params_to_web(p: MarketParams) -> dict[str, Any]:
    return {
        "regimes": list(p.regimes),
        "P": p.P.tolist(),
        "init_probs": p.initial_probs().tolist(),
        "equity": {"mu": p.equity.mu.tolist(), "sigma": p.equity.sigma.tolist()},
        "house": {"mu": p.house.mu.tolist(), "sigma": p.house.sigma.tolist(), "ar1": p.house_ar1},
        "levels": {v: {"theta": p.levels[v].theta.tolist(), "sigma": p.levels[v].sigma.tolist(),
                       "half_life_months": p.levels[v].half_life_months, "initial": p.levels[v].initial}
                   for v in LEVEL_VARS},
        "chol": cholesky_factors(p.corr).tolist(),
        "implied_vol": {"vrp_ratio": p.implied_vol.vrp_ratio, "skew": p.implied_vol.skew,
                        "min_vol": p.implied_vol.min_vol},
    }


def policy_to_web(year: int = 2026) -> dict[str, Any]:
    pol = load_policy(year)
    fname = pol["mortgage_income_test"]["financing_cost_table"]["file"]
    with resources.files("homepiggy.data.nl").joinpath(fname).open("r", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))[1:]
    g = pol["mortgage_income_test"]["student_loan_grossup"]
    return {
        "country": "nl",
        "year": year,
        "flp_income_rows": [float(r[0]) for r in rows],
        "flp_pct": [[float(x) / 100.0 for x in r[1:]] for r in rows],
        "flp_rate_upper_pct": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
        "grossup_upper_pct": [float(x) if x != float("inf") else 1e9 for x in g["brackets_upper_pct"]],
        "grossup_factors": [float(x) for x in g["factors"]],
        "annuity_term_months": int(pol["mortgage_income_test"]["annuity_term_months"]),
        "test_rate": pol["mortgage_income_test"]["test_rate"],
        "single_person_extra_loan": pol["mortgage_income_test"]["single_person_extra_loan"],
        "energy_label_extra_loan_eur": pol["mortgage_income_test"]["energy_label_extra_loan_eur"]["table"],
        "max_ltv": pol["loan_to_value"]["max_ltv"],
        "transfer_tax": pol["transfer_tax"],
        "nhg": pol["nhg"],
        "box3": pol["box3"],
        "sources": {
            "financing_cost_table": pol["mortgage_income_test"]["financing_cost_table"]["source"],
            "transfer_tax": pol["transfer_tax"]["source"], "nhg": pol["nhg"]["source"], "box3": pol["box3"]["source"],
        },
    }


def policy_uk_to_web(year: int = 2026) -> dict[str, Any]:
    pol = load_policy_uk(year)
    sd, L = pol["stamp_duty_land_tax"], pol["lifetime_isa"]
    f = sd["first_time_buyer_relief"]
    return {
        "country": "uk", "year": year, "currency": "GBP",
        "max_ltv": pol["lending"]["max_ltv"], "lti": pol["lending"]["loan_to_income_limit"],
        "sdlt": {"upper": [INF if x == float("inf") else float(x) for x in sd["standard_bands"]["upper_gbp"]],
                 "rates": [float(x) for x in sd["standard_bands"]["rates"]],
                 "ftb": {"nil_rate_upper": f["nil_rate_upper_gbp"], "reduced_rate": f["reduced_rate"],
                         "reduced_rate_upper": f["reduced_rate_upper_gbp"], "max_price": f["max_price_gbp"]}},
        "lisa": {"annual_limit": L["annual_limit_gbp"], "bonus_rate": L["bonus_rate"], "open_age_min": L["open_age_min"],
                 "open_age_max_exclusive": L["open_age_max_exclusive"],
                 "contribute_until_age_exclusive": L["contribute_until_age_exclusive"],
                 "price_threshold": L["property_price_threshold_gbp"],
                 "charge_above_threshold": L["withdrawal_charge_above_threshold"]},
        "isa_allowance": pol["isa"]["annual_allowance_gbp"],
        "sources": {"sdlt": sd["source"], "lisa": L["withdrawal_source"], "lending_ltv": pol["lending"]["max_ltv_source"],
                    "lending_lti": pol["lending"]["lti_source"], "isa": pol["isa"]["source"]},
    }


def policy_de_to_web(year: int = 2026) -> dict[str, Any]:
    pol = load_policy_de(year)
    ft = FundTax.from_policy(pol, "fund")
    it = dict(pol["income_tax"])
    return {
        "country": "de", "year": year, "currency": "EUR",
        "transfer_tax": {"rates": pol["transfer_tax"]["rates"], "names": pol["transfer_tax"]["names"],
                         "default_state": pol["transfer_tax"]["default_state"]},
        "notary_land_registry": {k: pol["notary_land_registry"][k] for k in ("notary_rate", "notary_vat", "land_registry_rate")},
        "lending": {k: pol["lending"][k] for k in ("max_ltv", "payment_share_of_net_income", "initial_repayment_rate")},
        "income_tax": {k: it[k] for k in ("zones_upper_eur", "zone2", "zone3", "zone4", "zone5", "employee_lump_sum_eur",
                                          "special_expenses_lump_sum_eur")},
        "solidarity_surcharge": {k: pol["solidarity_surcharge"][k] for k in ("rate", "exemption_limit_single_eur", "phase_in_rate")},
        "social_insurance": {k: v for k, v in pol["social_insurance"].items() if isinstance(v, (int, float))},
        "fund_tax": {k: getattr(ft, k) for k in ("rate", "allowance", "base_rate", "factor", "tf_equity", "tf_mixed",
                                                 "th_equity", "th_mixed")},
        "sources": {"transfer_tax": pol["transfer_tax"]["source"], "notary": pol["notary_land_registry"]["source"],
                    "income_tax": pol["income_tax"]["source"], "fund_tax": pol["investment_fund_tax"]["source"],
                    "base_rate": pol["investment_fund_tax"]["source_base_rate"], "lending": pol["lending"]["rule_source"]},
    }


def glidepaths_to_web(country: str = "nl") -> dict[str, Any]:
    """The optimised glidepath and the evidence behind it, from studies/output/glidepath_opt[_uk].json."""
    tag = "" if country == "nl" else f"_{country}"
    res = json.loads((ROOT / "studies" / "output" / f"glidepath_opt{tag}.json").read_text())
    if res.get("winner_index") is None:
        raise SystemExit("The optimisation found no glidepath passing the safety bar; nothing to export.")
    w = res["winner_index"]
    h = {m: res["heldout"][m]["rows"] for m in ("repo", "hmm")}
    win = {m: h[m][w] for m in h}
    today = {m: h[m][0] for m in h}
    bank = {m: h[m][-1] for m in h}
    crisis = res.get("crisis", {})
    n_crisis = sum(len(v) for v in crisis.values())
    n_pass = sum(1 for v in crisis.values() for e in v.values() if e["rows"][1]["safety_minus_bank_median"] >= 0)
    c = res["winner"]
    cash = f"cash for the last {c['c']} year{'s' if c['c'] != 1 else ''}" if c["c"] > 0 else "no separate cash phase"
    summary = (f"Global equity until {c['a']} years before buying, protected equity until {c['b']}, then short bonds"
               f"{'' if c['c'] == 0 else ' until ' + str(c['c'])}, with {cash}"
               + (f" (each step keeps {c['blend']:.0%} in the previous sleeve)" if c["blend"] else "") + ".")
    n_c = len(res["candidates"])
    R = res["setup"]["methods"]["repo"]["n_replicates"]
    how = [
        f"We tested {n_c} glidepaths that hand over from global equity to protected equity, short bonds and cash at different "
        f"times, on the same simulated futures, across {R} re-calibrations under both regime methods.",
        "Growth is the median pot divided by the deposit target. Safety is the average of that ratio in the worst 10% of "
        "futures (expected shortfall). The rule was fixed before running: a glidepath must be at least as safe as a "
        "savings account under both regime methods, and among those the highest growth wins. It was chosen on half of "
        "the re-calibrations and checked on the other half.",
        f"Checked on the held-out half (indicator-classifier regimes): growth {win['repo']['growth']:.2f} against "
        f"{today['repo']['growth']:.2f} for the original glidepath; safety {win['repo']['safety']:.2f} against "
        f"{bank['repo']['safety']:.2f} for a savings account, at least as safe in "
        f"{win['repo']['share_safer_than_bank']:.0%} of re-calibrations. HMM regimes: growth {win['hmm']['growth']:.2f} "
        f"vs {today['hmm']['growth']:.2f}, safety {win['hmm']['safety']:.2f} vs {bank['hmm']['safety']:.2f}. "
        f"In the crisis replays it stays at least as safe as a savings account in {n_pass} of {n_crisis} cases.",
    ]
    from homepiggy.glidepath_opt import A_YEARS, B_YEARS, BLENDS
    n_bar = sum(1 for m in ("repo", "hmm") for r in res["selection"][m]["rows"][1:-1] if r["passes_safety_bar"]) // 2
    edge = [txt for cond, txt in ((c["a"] == min(A_YEARS), "the latest equity handover tested"),
                                  (c["b"] == min(B_YEARS), "the latest bond handover tested"),
                                  (c["blend"] == max(BLENDS), "the largest blend tested")) if cond]
    if n_bar == n_c:
        how.append(f"Every one of the {n_c} glidepaths passed the safety bar, so growth decided."
                   + (f" The winner sits at the edge of what we tested ({', '.join(edge)}), so a later or softer "
                      "handover might do better still; that was not tested." if edge else ""))
    return {"optimised": {"label": c["label"], "table": c["table"], "summary": summary, "how": how,
                          "params": {k: c[k] for k in ("a", "b", "c", "blend")}},
            "run_date": res["setup"]["run_date"], "source": f"studies/output/glidepath_opt{tag}.json",
            "heldout": {m: {k: {kk: res["heldout"][m]["rows"][j][kk] for kk in ("growth", "safety", "p_target", "p_all3",
                                                                                "p_income_test", "share_safer_than_bank",
                                                                                "share_growth_above_today")}
                            for k, j in (("opt", w), ("today", 0), ("bank", -1))} for m in ("repo", "hmm")},
            "crisis_pass": [n_pass, n_crisis]}


def main(country: str = "nl") -> None:
    out = OUT / country
    out.mkdir(parents=True, exist_ok=True)
    for method, path in BOOT[country].items():
        params, prov = load_bootstrap(path)
        doc = {
            "method": method,
            "source_file": str(path.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "provenance": {k: prov[k] for k in ("regime_method", "description", "sample", "calibrated_utc", "sources")
                           if k in prov},
            "replicates": [params_to_web(p) for p in params],
        }
        (out / f"params_{method}.json").write_text(json.dumps(doc, separators=(",", ":")))
    pol = {"nl": policy_to_web, "uk": policy_uk_to_web, "de": policy_de_to_web}[country]()
    (out / "policy_2026.json").write_text(json.dumps(pol, indent=1, default=str))
    sc = json.loads((ROOT / "params" / f"{country}_scenarios.json").read_text())
    (out / "scenarios.json").write_text(json.dumps(sc, indent=1))
    tag = "" if country == "nl" else f"_{country}"
    if (ROOT / "studies" / "output" / f"glidepath_opt{tag}.json").exists():
        (out / "glidepaths.json").write_text(json.dumps(glidepaths_to_web(country), indent=1))
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", choices=("nl", "uk", "de"), default="nl")
    main(ap.parse_args().country)
