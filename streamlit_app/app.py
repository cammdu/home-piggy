"""home-piggy · glidepath lab (Streamlit). Run:  PYTHONPATH=. .venv/bin/streamlit run streamlit_app/app.py

Analysis app for the NL first-time-buyer glidepath fund. Every number comes from the
homepiggy engine and the calibrated, sourced parameter files. Results are option-C
bands across bootstrap re-calibrations; path-level charts show the median re-calibration.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import yaml  # noqa: E402

from homepiggy.affordability_nl import (  # noqa: E402
    BuyerProfile, FinancingCostTable, affordability_gap, load_policy,
)
from homepiggy.analysis import band_study, path_detail, subset  # noqa: E402
from homepiggy.engine import RunConfig, SaverProfile  # noqa: E402
from homepiggy.regimes import expected_durations_months  # noqa: E402
from homepiggy.scenarios import BASELINE, historical_scenario  # noqa: E402
from homepiggy.sleeves import ProductParams  # noqa: E402
from homepiggy.uncertainty import load_bootstrap  # noqa: E402
from streamlit_app import charts  # noqa: E402
from streamlit_app.presets import (  # noqa: E402
    METHODS, PARAMS, PRESETS, SLEEVE_NAMES, build_glidepaths, default_table_rows, load_events,
)


METRICS = {
    "p_reach_own_target": ("P(reach own target)", ".0%"),
    "p_reach_deposit": ("P(reach legal minimum)", ".0%"),
    "p_outpace_cash": ("P(outpace cash)", ".0%"),
    "p_keep_up_with_housing": ("P(keep up with house prices)", ".0%"),
    "p_all_three": ("P(legal minimum + cash + housing)", ".0%"),
    "pot_nominal.p50": ("Median pot at purchase", ",.0f"),
    "required_deposit.p50": ("Median legal minimum deposit", ",.0f"),
    "own_target.p50": ("Median own target", ",.0f"),
}


# ------------------------------------------------------------------ cached data
@st.cache_resource(show_spinner=False)
def bootstrap(suffix: str):
    return load_bootstrap(PARAMS / f"nl_bootstrap{suffix}.json")


@st.cache_resource(show_spinner=False)
def policy_and_table():
    pol = load_policy(2026)
    return pol, FinancingCostTable.load(pol)


@st.cache_data(show_spinner=False, max_entries=24)
def run_study(spec_json: str) -> dict:
    spec = json.loads(spec_json)
    params_all, prov = bootstrap(spec["suffix"])
    params = subset(params_all, spec["replicates"])
    saver = SaverProfile(**spec["saver"], buyer=BuyerProfile(**spec["buyer"]))
    product = ProductParams(**spec["product"])
    gps = build_glidepaths(spec["strategies"], spec["custom_rows"])
    scen = BASELINE
    if spec["scenario"] != "Baseline (no forced episode)":
        ev = next(e for e in load_events(spec["suffix"]) if e["name"] == spec["scenario"])
        scen = historical_scenario(ev, saver.horizon_years * 12, spec["scenario_end_before"])
    cfg = RunConfig(market=params[0], saver=saver, product=product, scenario=scen, seed=spec["seed"])
    band = band_study(cfg, gps, params, n_paths=spec["paths"])
    band.pop("per_replicate", None)
    detail = path_detail(cfg, gps, params[band["median_replicate"]], n_paths=max(2000, spec["paths"]))
    return {"band": band, "detail": detail, "sample": prov.get("sample"), "calibrated": prov.get("calibrated_utc")}


# ------------------------------------------------------------------ sidebar inputs
def main() -> None:
    st.set_page_config(page_title="home-piggy · glidepath lab", page_icon="🏠", layout="wide")
    st.sidebar.title("🏠 home-piggy")
    st.sidebar.caption("NL first-time-buyer glidepath lab · numbers only, every input sourced or yours")

    with st.sidebar.form("inputs"):
        st.subheader("Saver")
        horizon = st.slider("Years until purchase", 5, 12, 10)
        age = st.number_input("Age today", 18, 40, 20)
        income = st.number_input("Gross income today (€/yr)", 0, 500_000, 60_000, step=1_000)
        wage = st.slider("Real wage growth above inflation (%/yr)", 0.0, 5.0, 1.0, 0.25) / 100
        house = st.number_input("Target home's price today (€)", 50_000, 2_000_000, 400_000, step=10_000)
        monthly = st.number_input("Monthly contribution today (€)", 0, 10_000, 550, step=25)
        initial = st.number_input("Starting lump sum (€)", 0, 1_000_000, 1_000, step=500)
        own_pct = st.slider("Own savings target (% of price + costs; 0 = legal minimum only)", 0, 30, 10) / 100
        single = st.toggle("Buying alone (singles allowance)", value=True)
        with st.expander("More buyer details"):
            fixed = st.selectbox("Mortgage fixed period", [10, 20, 30, 5], index=0,
                                 help="Under 10 years the test rate is at least 5% (AFM).")
            debt = st.number_input("DUO student debt (€)", 0, 200_000, 0, step=1_000)
            nhg = st.toggle("Use NHG (0.4% premium if eligible)", value=True)
            other_costs = st.number_input("Other buying costs: notary, valuation, advice (€, 2026 prices)", 0, 50_000, 0,
                                          step=500)

        st.subheader("Strategies")
        strategies = st.multiselect("Glidepaths to compare (first = baseline for paired differences)",
                                    list(PRESETS) + ["Custom table"],
                                    default=["Agreed step glidepath", "Late de-risk (γ = 0.6)", "Lock in when target reached"])
        st.caption("Custom table (used only if 'Custom table' is selected; weights in %, each row must sum to 100)")
        custom = st.data_editor(pd.DataFrame(default_table_rows()), num_rows="dynamic", hide_index=True,
                                key="custom_tbl")

        st.subheader("Protected equity")
        lower = st.slider("Buffer: losses absorbed (%)", 5, 20, 10, 1)
        funding = st.slider("Call overwrite funding share (0 = pure put spread)", 0.0, 1.0, 0.0, 0.05)

        st.subheader("Uncertainty & scenario")
        method = st.radio("Regime method", list(METHODS) + ["Both, side by side"], index=2)
        replicates = st.select_slider("Re-calibrations per method", [25, 50, 100, 200], value=50)
        paths = st.select_slider("Paths per re-calibration", [250, 500, 1000, 2000], value=500)
        events = ["Baseline (no forced episode)"] + [e["name"] for e in load_events("")]
        scenario = st.selectbox("Historical episode", events)
        end_before = st.slider("Episode ends … months before purchase", 0, 96, 0)
        seed = st.number_input("Random seed", 1, 2**31 - 1, 20261110)
        submitted = st.form_submit_button("Run analysis", type="primary", width="stretch")

    saver = {"horizon_years": int(horizon), "initial_eur": float(initial), "monthly_contribution_eur": float(monthly),
             "gross_income_eur": float(income), "real_wage_growth": float(wage), "house_price_eur": float(house),
             "student_debt_eur": float(debt), "own_target_pct": float(own_pct)}
    buyer = {"age_at_start": float(age), "single": bool(single), "fixed_period_years": int(fixed), "use_nhg": bool(nhg),
             "other_costs_eur": float(other_costs)}
    product = {"buffer_lower_strike": 1.0 - lower / 100, "call_funding": float(funding)}
    methods = list(METHODS.values()) if method.startswith("Both") else [METHODS[method]]
    method_names = {v: k for k, v in METHODS.items()}

    # ------------------------------------------------------------------ header
    st.title("Glidepath lab: Dutch first-time buyers")
    st.markdown(
        "Results are **bands**: the median and 5th–95th percentile across bootstrap **re-calibrations** of the model "
        "(parameter uncertainty, agreed option C). Path charts show the **median re-calibration** only. "
        "Read **Methods & limitations** before using any number.")

    problems = []
    if not strategies:
        problems.append("Pick at least one glidepath.")
    if scenario != events[0]:
        ev = next(e for e in load_events("") if e["name"] == scenario)
        if ev["duration_months"] + end_before > horizon * 12:
            problems.append(f"Episode ({ev['duration_months']} months) ending {end_before} months before purchase "
                            f"doesn't fit in {horizon} years.")
    if problems:
        for p in problems:
            st.error(p)
        st.stop()

    if "results" not in st.session_state and not submitted:
        st.info("Set your inputs in the sidebar and press **Run analysis**. With the defaults (both regime methods, "
                "3 strategies, 50 re-calibrations × 500 paths) a run takes about 30–60 seconds; results are cached.")
        st.stop()
    if submitted:
        st.session_state.results = {}
        for suf in methods:
            spec = {"suffix": suf, "replicates": int(replicates), "paths": int(paths), "seed": int(seed),
                    "saver": saver, "buyer": buyer, "product": product, "strategies": strategies,
                    "custom_rows": custom.to_dict("records"), "scenario": scenario, "scenario_end_before": int(end_before)}
            with st.spinner(f"Running {len(strategies)} strategies × {replicates} re-calibrations × {paths} paths "
                            f"({method_names[suf]})…"):
                try:
                    st.session_state.results[suf] = run_study(json.dumps(spec, sort_keys=True))
                except ValueError as exc:
                    st.error(f"Input rejected by the model: {exc}")
                    st.stop()
    results = st.session_state.results
    first = next(iter(results.values()))

    tabs = st.tabs(["Results", "Paths", "Affordability", "Drivers", "Regimes", "Methods & limitations"])

    # ------------------------------------------------------------------ Results
    with tabs[0]:
        st.caption(f"Calibration sample {first['sample']} · {first['band']['n_replicates']} re-calibrations × "
                   f"{first['band']['n_paths']} paths per method · scenario: {scenario}")
        metric = st.selectbox("Outcome", list(METRICS), format_func=lambda k: METRICS[k][0])
        label, fmt = METRICS[metric]
        rows = []
        for suf, res in results.items():
            for strat, b in res["band"]["bands"].items():
                v = b[metric]
                rows.append({"strategy": strat, "method": method_names[suf], "median": v["median"], "p05": v["p05"],
                             "p95": v["p95"], "mc_noise_ratio": v["mc_noise_ratio"]})
        df = pd.DataFrame(rows)
        st.altair_chart(charts.interval_chart(df, value_fmt=".0%" if fmt == ".0%" else "~s", title=label),
                        width="stretch")
        noisy = df[df["mc_noise_ratio"] > 0.25]
        if len(noisy):
            st.warning("Simulation noise is large relative to some bands (ratio > 0.25). Increase paths per re-calibration.")

        st.markdown("**All outcomes:** median [5th–95th percentile across re-calibrations]")
        table = []
        for suf, res in results.items():
            for strat, b in res["band"]["bands"].items():
                row = {"method": method_names[suf], "strategy": strat}
                for k, (lab, f) in METRICS.items():
                    v = b[k]
                    row[lab] = (f"{v['median']:.0%} [{v['p05']:.0%}–{v['p95']:.0%}]" if f == ".0%" else
                                f"€{v['median']:,.0f} [€{v['p05']:,.0f}–€{v['p95']:,.0f}]")
                table.append(row)
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")

        for suf, res in results.items():
            pv = res["band"]["paired_vs_first"]
            if pv:
                st.markdown(f"**Paired comparison vs '{res['band']['baseline']}'** ({method_names[suf]}): change in "
                            "success rate on identical paths, per re-calibration")
                st.dataframe(pd.DataFrame([{
                    "strategy": s, "kind": "own target" if k == "target" else "legal minimum",
                    "median change": f"{v['median']:+.1%}", "5–95% band": f"{v['p05']:+.1%} to {v['p95']:+.1%}",
                    "better in": f"{v['share_of_replicates_better']:.0%} of re-calibrations",
                    "worse in": f"{v['share_worse']:.0%}"} for s, d in pv.items() for k, v in d.items()]),
                    hide_index=True, width="stretch")

    # ------------------------------------------------------------------ Paths
    with tabs[1]:
        suf = st.radio("Method", list(results), format_func=lambda s: method_names[s], horizontal=True, key="paths_m") \
            if len(results) > 1 else next(iter(results))
        det = results[suf]["detail"]
        strat = st.selectbox("Strategy", list(det), key="paths_s")
        real = st.toggle("Show pot in 2026 prices (inflation-adjusted)", value=False)
        st.caption("Median re-calibration only. Blue: pot, 5–95% and 25–75% of paths, median line. Orange: your target "
                   "(max of legal minimum and own goal), median and 5–95% dashed.")
        st.altair_chart(charts.fan_chart(det, strat, real=real), width="stretch")
        st.markdown("**Average allocation over time**")
        st.altair_chart(charts.allocation_chart(det, strat), width="stretch")
        s = det[strat]["summary"]
        st.dataframe(pd.DataFrame([
            {"quantity": "Pot at purchase", **{q: f"€{s['pot_nominal'][q]['value']:,.0f}" for q in s["pot_nominal"]}},
            {"quantity": "Your target", **{q: f"€{s['own_target'][q]['value']:,.0f}" for q in s["own_target"]}},
            {"quantity": "Max drawdown, last 24 months", **{q: f"{s['max_drawdown_last_24m'][q]['value']:.1%}"
                                                           for q in s["max_drawdown_last_24m"]}},
            {"quantity": "Money-weighted return p.a.", **{q: f"{s['money_weighted_return_pa'][q]['value']:.1%}"
                                                          for q in s["money_weighted_return_pa"]}},
        ]), hide_index=True, width="stretch")
        st.caption(f"Box 3 tax paid (mean): €{s['box3_tax_mean_eur']:,.0f} · trading costs (mean): €{s['costs_mean_eur']:,.0f}"
                   f" · contributions (mean): €{s['total_contributed_mean_eur']:,.0f}")

    # ------------------------------------------------------------------ Affordability
    with tabs[2]:
        pol, tab = policy_and_table()
        ref = yaml.safe_load((PARAMS / "nl_calibrated.yaml").read_text())["market"]
        m_now = ref["mortgage_rate"]["initial"]
        b = BuyerProfile(**buyer)
        a = affordability_gap(np.array([house]), np.array([income]), np.array([m_now]),
                              np.array([0.0]), age, np.array([1.0]), np.array([1.0]), b, pol, tab)
        st.markdown(f"**If you bought today** (NL 2026 rules, mortgage rate {m_now:.2%} = latest ECB MIR, >10y fixed)")
        c = st.columns(4)
        c[0].metric("Max mortgage", f"€{a.max_loan[0]:,.0f}", help="Lower of the income test and 100% of value")
        c[1].metric("Multiple of income", f"{a.max_loan[0] / max(income, 1):.2f}×")
        c[2].metric("Buying costs", f"€{(a.transfer_tax + a.nhg_premium + a.other_costs)[0]:,.0f}")
        c[3].metric("Own money needed (legal minimum)", f"€{a.required_deposit[0]:,.0f}")
        st.caption(f"Test rate {a.test_rate[0]:.2%} · financing-cost share {a.flp[0]:.1%} of income "
                   "(Staatscourant 2025/36471) · singles allowance and student-debt gross-up applied as set.")
        suf = st.radio("Method", list(results), format_func=lambda s: method_names[s], horizontal=True, key="aff_m") \
            if len(results) > 1 else next(iter(results))
        det = results[suf]["detail"]
        strat = st.selectbox("Strategy", list(det), key="aff_s")
        ap = det[strat]["at_purchase"]
        st.markdown("**At the purchase date** (median re-calibration; 5th / 50th / 95th percentile of paths)")
        names = {"price": "House price", "income": "Gross income", "mortgage_rate": "Mortgage rate", "test_rate": "Test rate",
                 "max_loan": "Max mortgage", "transfer_tax": "Transfer tax", "nhg": "NHG premium",
                 "legal_min": "Legal minimum own funds", "own_target": "Your target", "pot": "Pot"}
        st.dataframe(pd.DataFrame([{"item": names[k], **{q: (f"{ap[k][q]:.2%}" if "rate" in k else f"€{ap[k][q]:,.0f}")
                                                         for q in ("p05", "p50", "p95")}} for k in names]),
                     hide_index=True, width="stretch")

    # ------------------------------------------------------------------ Drivers
    with tabs[3]:
        st.markdown("Which **estimated parameters** move the result across re-calibrations (Spearman rank correlation "
                    "between the parameter and the strategy's P(reach own target)). Positive = higher parameter, higher "
                    "success.")
        for suf, res in results.items():
            for strat, b in res["band"]["bands"].items():
                d = b["p_reach_own_target"]["drivers_spearman"]
                if d:
                    st.markdown(f"**{strat}** · {method_names[suf]}")
                    st.altair_chart(charts.drivers_chart(d), width="stretch")
                    break

    # ------------------------------------------------------------------ Regimes
    with tabs[4]:
        rg = pd.read_csv(ROOT / "calibration" / "output" / "regimes.csv")
        regimes = ["Expansion", "Late-cycle", "Recession", "Recovery"]
        long = pd.concat([
            pd.DataFrame({"month": pd.to_datetime(rg["month"]), "method": "Repo classifier", "regime": rg["regime_repo"]}),
            pd.DataFrame({"month": pd.to_datetime(rg["month"]), "method": "HMM", "regime": rg["regime_hmm"]})])
        st.markdown("**Historical regime labels, 2002–2026** (euro-area indicators)")
        st.altair_chart(charts.regime_strip(long, regimes), width="stretch")
        cols = st.columns(2)
        for col, (name, suf) in zip(cols, METHODS.items()):
            y = yaml.safe_load((PARAMS / f"nl_calibrated{suf}.yaml").read_text())["market"]
            P = np.array(y["transition_matrix"])
            with col:
                st.markdown(f"**{name}**")
                st.dataframe(pd.DataFrame({
                    "regime": regimes, "probability today": [f"{p:.0%}" for p in y["initial_regime"]],
                    "expected spell (months)": [f"{d:.1f}" for d in expected_durations_months(P)],
                    "house growth p.a.": [f"{x:.1%}" for x in y["house"]["mu"]],
                    "mortgage level": [f"{x:.1%}" for x in y["mortgage_rate"]["theta"]],
                    "equity return p.a.": [f"{x:.1%}" for x in y["equity"]["mu"]]}),
                    hide_index=True, width="stretch")
                st.caption("Monthly transition matrix (rows = from)")
                st.dataframe(pd.DataFrame(P, index=regimes, columns=regimes).style.format("{:.2f}"),
                             width="stretch")

    # ------------------------------------------------------------------ Docs
    with tabs[5]:
        doc = st.radio("Document", ["LIMITATIONS", "METHODS", "CALIBRATION", "SOURCES"], horizontal=True)
        st.markdown((ROOT / "docs" / f"{doc}.md").read_text())


# Streamlit executes this file as __main__. Worker processes started by band_study (spawn) import it as
# __mp_main__ and must NOT re-run the UI, otherwise every worker would launch the app again.
if __name__ == "__main__":
    main()
