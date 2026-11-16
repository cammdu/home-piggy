"""Export the calibrated parameters, NL rules and scenario library as JSON for the browser app (web/).

The browser engine (web/engine.js) reads exactly these files. Two things are precomputed here so the
JavaScript port needs no linear algebra: the Cholesky factor of each regime's (PSD-repaired)
correlation matrix, via the same `cholesky_factors` used by the Python engine.

Usage: PYTHONPATH=. python tools/export_web_data.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any

from homepiggy.affordability_nl import load_policy
from homepiggy.config import LEVEL_VARS, MarketParams
from homepiggy.market import cholesky_factors
from homepiggy.uncertainty import load_bootstrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "data"
BOOT = {"repo": ROOT / "params" / "nl_bootstrap.json", "hmm": ROOT / "params" / "nl_bootstrap_hmm.json"}


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


def glidepaths_to_web() -> dict[str, Any]:
    """The optimised glidepath and the evidence behind it, from studies/output/glidepath_opt.json."""
    res = json.loads((ROOT / "studies" / "output" / "glidepath_opt.json").read_text())
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
    summary = (f"Shares until {c['a']} years before buying, buffered shares until {c['b']}, then short bonds"
               f"{'' if c['c'] == 0 else ' until ' + str(c['c'])}, with {cash}"
               + (f" (each step keeps {c['blend']:.0%} in the previous sleeve)" if c["blend"] else "") + ".")
    n_c = len(res["candidates"])
    R = res["setup"]["methods"]["repo"]["n_replicates"]
    how = [
        f"We tested {n_c} glidepaths that hand over from shares to buffered shares, short bonds and cash at different "
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
            "run_date": res["setup"]["run_date"], "source": "studies/output/glidepath_opt.json"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for method, path in BOOT.items():
        params, prov = load_bootstrap(path)
        doc = {
            "method": method,
            "source_file": str(path.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "provenance": {k: prov[k] for k in ("regime_method", "description", "sample", "calibrated_utc", "sources")
                           if k in prov},
            "replicates": [params_to_web(p) for p in params],
        }
        (OUT / f"params_{method}.json").write_text(json.dumps(doc, separators=(",", ":")))
    (OUT / "policy_nl_2026.json").write_text(json.dumps(policy_to_web(), indent=1, default=str))
    sc = json.loads((ROOT / "params" / "nl_scenarios.json").read_text())
    (OUT / "scenarios.json").write_text(json.dumps(sc, indent=1))
    if (ROOT / "studies" / "output" / "glidepath_opt.json").exists():
        (OUT / "glidepaths.json").write_text(json.dumps(glidepaths_to_web(), indent=1))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
