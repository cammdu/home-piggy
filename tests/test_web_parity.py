"""Parity: the browser engine (web/engine.js) must reproduce the Python engine path by path.

numpy's random draws are recorded and replayed through the JS engine under node, so any difference is
a porting error, not sampling noise. Covers both regime methods, a historical replay (forced regimes +
shocks), student debt, a short fixed period (5% test-rate floor), the starters-exemption age limit and
an all-equity glidepath. Tolerance: relative 1e-9 (floating-point order of operations and libm ulps).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from homepiggy.affordability_nl import BuyerProfile, FinancingCostTable, affordability_gap, load_policy
from homepiggy.engine import RunConfig, SaverProfile, _streams, _student_instalment, run
from homepiggy.metrics import irr_monthly
from homepiggy.glidepath import DEFAULT_TABLE, GlidepathSpec
from homepiggy.scenarios import historical_scenario
from homepiggy.uncertainty import load_bootstrap
from tools.export_web_data import params_to_web, policy_to_web

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

ALL_EQUITY = [{"from_years": 15, "to_years": 0, "weights": [1.0, 0.0, 0.0, 0.0]}]
CASES = [
    dict(name="repo_baseline_default", method="nl_bootstrap", rep=0, saver={}, buyer={}, table=None, event=None, end=0),
    dict(name="repo_income_test_on", method="nl_bootstrap", rep=0, saver=dict(apply_income_test=True), buyer={},
         table=None, event=None, end=0),
    dict(name="hmm_gfc_student_short_fix", method="nl_bootstrap_hmm", rep=7,
         saver=dict(house_price_eur=310000.0, student_debt_eur=25000.0, gross_income_eur=48000.0, apply_income_test=True),
         buyer=dict(fixed_period_years=5, age_at_start=28.0, energy_label="AB", other_costs_eur=4000.0),
         table=None, event="gfc_2008", end=0),
    dict(name="repo_all_equity_covid", method="nl_bootstrap", rep=123, saver=dict(horizon_years=8, monthly_contribution_eur=800.0),
         buyer={}, table=ALL_EQUITY, event="covid_2020", end=12),
]
N_PATHS = 40


def _case(c: dict, scen_events: dict) -> tuple[dict, dict]:
    params = load_bootstrap(ROOT / "params" / f"{c['method']}.json")[0][c["rep"]]
    buyer = replace(BuyerProfile(), **c["buyer"])
    saver = replace(SaverProfile(), **c["saver"], buyer=buyer)
    T = saver.horizon_years * 12
    table = c["table"] or [dict(r) for r in DEFAULT_TABLE]
    key = "hmm" if c["method"].endswith("hmm") else "repo_classifier"
    event = next(e for e in scen_events[key] if e["name"] == c["event"]) if c["event"] else None
    kw = {}
    if event:
        kw["scenario"] = historical_scenario(event, T, c["end"])
    cfg = RunConfig(market=params, saver=saver, glidepath=GlidepathSpec(kind="step", table=table), n_paths=N_PATHS,
                    seed=4242 + c["rep"], **kw)
    gi, gt, gn = _streams(cfg.seed)                      # the exact draws simulate_market_paths makes
    draws = {"init": gi.random(N_PATHS).tolist(), "U": gt.random((N_PATHS, T)).ravel().tolist(),
             "Z": gn.standard_normal((N_PATHS, T, 6)).ravel().tolist()}
    r = run(cfg)
    # measure 2: own funds needed with the income test, and money-weighted returns (metrics.irr_monthly)
    pol = load_policy(cfg.policy_year)
    a = affordability_gap(r.price[:, -1], r.income[:, -1], r.market.mortgage_rate[:, -1],
                          np.full(N_PATHS, _student_instalment(saver, T)[T]), buyer.age_at_start + T / 12,
                          r.market.cpi[:, -1], r.market.house_index[:, -1], buyer, pol, FinancingCostTable.load(pol),
                          apply_income_test=True)
    g_inc = np.maximum(a.required_deposit, saver.own_target_pct * r.price[:, -1] + a.transfer_tax + a.nhg_premium
                       + a.other_costs)

    def irr(port):
        f = -port.contributions.copy(); f[:, -1] += port.value[:, -1]
        return (1.0 + irr_monthly(f)) ** 12 - 1.0
    s = {k: v for k, v in vars(saver).items() if k != "buyer"} | vars(buyer)
    fx = {"name": c["name"], "params": params_to_web(params), "saver": s, "table": table, "event": event,
          "end_months_before_purchase": c["end"], "n_paths": N_PATHS, "draws": draws, "mid_month": T // 2}
    exp = {"V": r.portfolio.value[:, -1], "B": r.benchmark.value[:, -1], "G": r.target_path[:, -1],
           "D": r.required_path[:, -1], "Vpath": r.portfolio.value, "G_mid": r.target_path[:, T // 2],
           "D_mid": r.required_path[:, T // 2], "equity_index": r.market.equity_index,
           "house_index": r.market.house_index, "mortgage_rate": r.market.mortgage_rate, "cpi": r.market.cpi,
           "regimes": r.market.regimes, "Dinc": a.required_deposit, "Ginc": g_inc, "irr": irr(r.portfolio),
           "bank_irr": irr(r.benchmark), "house_pa": r.market.house_index[:, -1] ** (12.0 / T) - 1.0}
    return fx, exp


@pytest.fixture(scope="module")
def parity(tmp_path_factory):
    scen = json.loads((ROOT / "params" / "nl_scenarios.json").read_text())["events"]
    fxs, exps = zip(*[_case(c, scen) for c in CASES])
    d = tmp_path_factory.mktemp("parity")
    (d / "fx.json").write_text(json.dumps({"policy": json.loads(json.dumps(policy_to_web(), default=str)),
                                           "cases": list(fxs)}))
    subprocess.run([NODE, str(ROOT / "web" / "test" / "parity_runner.mjs"), str(d / "fx.json"), str(d / "out.json")],
                   check=True, cwd=ROOT)
    got = json.loads((d / "out.json").read_text())
    return {g["name"]: (g, e) for g, e in zip(got, exps)}


@pytest.mark.parametrize("name", [c["name"] for c in CASES])
@pytest.mark.parametrize("key", ["regimes", "equity_index", "house_index", "mortgage_rate", "cpi",
                                 "Vpath", "V", "B", "G", "D", "G_mid", "D_mid", "Dinc", "Ginc",
                                 "irr", "bank_irr", "house_pa"])
def test_js_matches_python(parity, name, key):
    got, exp = parity[name]
    e = np.asarray(exp[key], float).ravel()
    g = np.asarray(got[key], float).ravel()
    assert g.shape == e.shape
    np.testing.assert_allclose(g, e, rtol=1e-9, atol=1e-6)


def test_cases_exercise_the_rules(parity):
    """Guard against a vacuous test: the cases must actually hit the branches they claim to."""
    got, _ = parity["hmm_gfc_student_short_fix"]
    assert np.ptp(np.asarray(got["regimes"])) > 0
    _, exp = parity["repo_baseline_default"]
    assert 0.0 < np.mean(exp["V"] >= exp["G"]) < 1.0      # success threshold is crossed on some paths
