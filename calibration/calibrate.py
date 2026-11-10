"""End-to-end calibration: data -> regimes -> estimates -> bootstrap -> files.

Run:  PYTHONPATH=. .venv/bin/python -m calibration.calibrate [--replicates 200] [--seed 20261110]

Outputs
  params/nl_calibrated.yaml            full-sample estimates (the bootstrap's reference set)
  params/nl_bootstrap.json             B re-calibrated parameter sets (option C)
  calibration/output/report.json       diagnostics: labels, cross-check, fallbacks, tests
  calibration/output/regimes.csv       month, regime label, softmax probabilities, HMM label
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from homepiggy.config import market_params_from_dict
from .bootstrap import politis_white_block_length, stationary_indices
from .classify import REGIMES, align_states, classify, cohens_kappa, dimension_scores, fit_hmm
from .dataset import EONIA_ESTR_SOURCE, build_panel
from .estimate import LEVELS, estimate
from .scenarios_hist import build_events
from .validate import pit_checks, ten_year_coverage
from .sources import REGISTRY

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "calibration" / "output"
SAMPLE_START = np.datetime64("1999-01", "M")
TENOR_1Y = 1.0


def _manifest() -> dict:
    p = OUT / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def house_seasonality(g: np.ndarray, months: np.ndarray) -> dict:
    """F-test of calendar-month dummies on monthly house growth."""
    ok = ~np.isnan(g)
    cal = months.astype(int) % 12
    groups = [g[ok & (cal == c)] for c in range(12)]
    F, p = stats.f_oneway(*groups)
    means = np.array([x.mean() for x in groups])
    return {"F": float(F), "p_value": float(p), "calendar_means": means}


def implied_vol_params(vix_var: np.ndarray, rv: np.ndarray, skew: np.ndarray, vix_min: float) -> dict:
    """vrp = sqrt(mean implied var_t / mean realised var_{t+1}); skew via Backus-Foresi-Wu (2004)."""
    ok = ~np.isnan(vix_var[:-1]) & ~np.isnan(rv[1:])
    vrp = float(np.sqrt(vix_var[:-1][ok].mean() / rv[1:][ok].mean()))
    s = skew[~np.isnan(skew)]
    gamma_30d = (100.0 - s.mean()) / 10.0                     # Cboe: SKEW = 100 - 10 * skewness
    gamma_1y = gamma_30d * np.sqrt(30.0 / 365.0)              # iid scaling of skewness: 1/sqrt(horizon)
    slope = -gamma_1y / (6.0 * np.sqrt(TENOR_1Y))             # sigma(K) ~ sigma + (gamma/6) ln(K/S)/sqrt(tau)
    return {"vrp_ratio": vrp, "skew": float(slope), "min_vol": float(vix_min),
            "_detail": {"skew_index_mean": float(s.mean()), "gamma_30d": float(gamma_30d), "gamma_1y": float(gamma_1y),
                        "months_vrp": int(ok.sum())}}


def assemble(est_params: dict, initial: dict, init_probs: np.ndarray, iv: dict) -> dict:
    m = {"regimes": list(REGIMES), "transition_matrix": est_params["transition_matrix"],
         "initial_regime": [float(x) for x in init_probs],
         "equity": est_params["equity"], "house": est_params["house"],
         "correlation": est_params["correlation"],
         "implied_vol": {k: iv[k] for k in ("vrp_ratio", "skew", "min_vol")}}
    for v in LEVELS:
        m[v] = {**est_params[v], "initial": float(initial[v])}
    return m


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main(replicates: int = 200, seed: int = 20261110) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    P = build_panel()
    months = P.months

    # ---------------------------------------------------------------- regimes (primary + cross-check)
    scores = dimension_scores(P.ind)
    labels, probs = classify(scores)
    in_sample = (months >= SAMPLE_START) & (labels >= 0) & ~np.isnan(P.v["house"])
    first = int(np.argmax(in_sample))
    last = int(len(months) - 1 - np.argmax(in_sample[::-1]))
    if not in_sample[first:last + 1].all():
        raise RuntimeError("Calibration sample is not contiguous; check data gaps.")
    S = slice(first, last + 1)
    sm = months[S]
    lab = labels[S]

    hmm = fit_hmm(scores[S], k=4, n_starts=20, seed=seed)
    hmm_lab, conf, mapping = align_states(hmm.states, lab)
    kappa_agree = cohens_kappa(conf)
    hmm_init = np.zeros(4)
    for st, r in mapping.items():
        hmm_init[r] += hmm.post[-1][st]       # filtered = smoothed probability at the last month

    # ---------------------------------------------------------------- data rows for estimation
    seas = house_seasonality(P.v["house"][S], sm)
    house = P.v["house"][S].copy()
    deseasonalised = seas["p_value"] < 0.01
    if deseasonalised:   # remove calendar-month means, keep the overall mean
        cal = sm.astype(int) % 12
        adj = seas["calendar_means"] - np.nanmean(house)
        house = house - adj[cal]
    rows = {"equity": P.v["equity"][S], "house": house}
    for v in LEVELS:
        rows[v] = P.v[v][S]

    initial = {}
    for v in LEVELS:
        y = P.v[v]
        i = np.where(~np.isnan(y))[0][-1]
        initial[v] = float(y[i])
        initial[f"{v}_month"] = str(months[i])
    aux = {k: P.aux[k][S] for k in ("vix_var", "realised_var", "cboe_skew")}
    iv = implied_vol_params(aux["vix_var"], aux["realised_var"], aux["cboe_skew"], float(P.aux["vix_min"][0]))

    manifest = _manifest()
    sources = [{"name": n, "provider": REGISTRY[n].provider, "key": REGISTRY[n].key, "url": REGISTRY[n].url,
                "retrieved_utc": manifest.get(n, {}).get("retrieved_utc"), "sha256": manifest.get(n, {}).get("sha256")}
               for n in REGISTRY]
    sources.append({"name": "eonia_estr_spread", "provider": "ECB", "url": EONIA_ESTR_SOURCE, "value_bp": 8.5})
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sample = f"{sm[0]} to {sm[-1]} ({len(sm)} months)"

    variants = {}
    rows_raw = {**rows, "house_raw": P.v["house"][S]}
    for method, labs, init_p, suffix in (("repo_classifier", lab, probs[last], ""),
                                         ("hmm", hmm_lab, hmm_init, "_hmm")):
        variants[method] = _variant(method, suffix, rows, labs, init_p, initial, iv, aux, replicates, seed,
                                    sources, stamp, sample, sm)
        variants[method]["report"]["validation"] = {
            "one_step_pit": pit_checks(rows, labs, variants[method]["ref"], variants[method]["house_z"]),
            "ten_year_coverage": ten_year_coverage(rows_raw, labs, variants[method]["ref"]),
        }

    # ---------------------------------------------------------------- historical scenarios (data-derived)
    rows_s = {**rows, "house_raw": P.v["house"][S]}
    scen = {"provenance": {"description": "Historical-replay episodes measured from the calibration data; windows "
                           "found algorithmically; shocks = observed move minus the calibrated model's expected path.",
                           "calibrated_utc": stamp, "sources": sources},
            "events": {m: build_events(rows_s, labs, sm, variants[m]["ref"], REGIMES)
                       for m, labs in (("repo_classifier", lab), ("hmm", hmm_lab))}}
    (ROOT / "params" / "nl_scenarios.json").write_text(json.dumps(_jsonable(scen), indent=1))

    report = {
        "calibrated_utc": stamp, "sample": sample,
        "house_seasonality": {"F": seas["F"], "p_value": seas["p_value"], "deseasonalised": bool(deseasonalised)},
        "implied_vol": iv, "initial": initial,
        "regime_cross_check": {"hmm_loglik": hmm.loglik, "agreement": float(np.trace(conf) / conf.sum()),
                               "cohens_kappa": kappa_agree, "confusion_repo_rows_hmm_cols": conf.tolist()},
        "variants": {m: v["report"] for m, v in variants.items()},
        "scenarios": {m: [{k: e[k] for k in ("name", "window", "duration_months", "observed_change", "shocks")}
                          for e in evs] for m, evs in scen["events"].items()},
        "panel_notes": P.notes,
    }
    (OUT / "report.json").write_text(json.dumps(_jsonable(report), indent=2))
    with open(OUT / "regimes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["month", "regime_repo", "regime_hmm"] + [f"p_{r}" for r in REGIMES] + ["s_g", "s_pi", "s_pol", "s_fc"])
        for i, mth in enumerate(sm):
            w.writerow([str(mth), REGIMES[lab[i]], REGIMES[hmm_lab[i]]] + [f"{x:.4f}" for x in probs[S][i]]
                       + [f"{x:.4f}" for x in scores[S][i]])
    return report


def _variant(method, suffix, rows, lab, init_probs, initial, iv, aux, replicates, seed, sources, stamp, sample, sm):
    vp = np.ones(len(sm), dtype=bool); vp[0] = False
    full = estimate(rows, lab, vp, REGIMES)
    ref = assemble(full.params, initial, init_probs, iv)
    market_params_from_dict(ref)        # fail closed if anything is out of range

    key_series = {
        "equity": rows["equity"], "house": rows["house"],
        "d_short_rate": np.r_[np.nan, np.diff(rows["short_rate"])],
        "d_mortgage_rate": np.r_[np.nan, np.diff(rows["mortgage_rate"])],
        "d_inflation": np.r_[np.nan, np.diff(rows["inflation"])],
        "recession_indicator": (lab == REGIMES.index("Recession")).astype(float),
    }
    block = {k: politis_white_block_length(v) for k, v in key_series.items()}
    L = max(block.values())
    rng = np.random.default_rng(seed)
    reps, fallback_log, invalid = [], [], []
    for b in range(replicates):
        idx, valid = stationary_indices(len(sm), L, rng)
        e = estimate({k: v[idx] for k, v in rows.items()}, lab[idx], valid, REGIMES, fallback=full.params)
        vv = aux["vix_var"][idx][:-1]                 # implied variance in month j
        rv_next = aux["realised_var"][idx][1:]        # realised variance in month j+1 (true successor only)
        ok = valid[1:] & ~np.isnan(vv) & ~np.isnan(rv_next)
        iv_b = dict(iv)
        if ok.sum() > 24:
            iv_b["vrp_ratio"] = float(np.sqrt(vv[ok].mean() / rv_next[ok].mean()))
        sk = aux["cboe_skew"][idx]; sk = sk[~np.isnan(sk)]
        iv_b["skew"] = float(-((100.0 - sk.mean()) / 10.0 * np.sqrt(30.0 / 365.0)) / 6.0)
        params_b = assemble(e.params, initial, init_probs, iv_b)
        try:
            market_params_from_dict(params_b)
        except ValueError as exc:
            invalid.append({"replicate": b, "error": str(exc)})
            continue
        reps.append(params_b)
        fallback_log.append(e.fallbacks)
    if invalid:
        raise RuntimeError(f"{method}: {len(invalid)} bootstrap replicates failed validation: {invalid[:3]} "
                           "(not dropped silently; investigate before use).")

    prov = {"synthetic": False, "regime_method": method,
            "description": "Full-sample calibration (reference set). Per the agreed method (option C), results are "
                           f"reported as a band across params/nl_bootstrap{suffix}.json, not from this file alone.",
            "sample": sample, "calibrated_utc": stamp, "seed": seed, "sources": sources}
    with open(ROOT / "params" / f"nl_calibrated{suffix}.yaml", "w") as fh:
        yaml.safe_dump({"provenance": prov, "market": _jsonable(ref)}, fh, sort_keys=False, width=120)
    with open(ROOT / "params" / f"nl_bootstrap{suffix}.json", "w") as fh:
        json.dump({"provenance": {**prov, "description": f"{len(reps)} stationary-bootstrap re-calibrations "
                                  f"(regime method: {method})", "mean_block_length_months": L},
                   "replicates": _jsonable(reps)}, fh)
    fb_counts: dict[str, int] = {}
    for fl in fallback_log:
        for f in set(fl):
            fb_counts[f] = fb_counts.get(f, 0) + 1
    P_ = np.array(ref["transition_matrix"])
    from homepiggy.regimes import expected_durations_months, stationary_distribution
    report = {
        "regime_months": full.diagnostics["months_per_regime"],
        "transition_counts": full.diagnostics["transition_counts"],
        "stationary_distribution": dict(zip(REGIMES, stationary_distribution(P_).tolist())),
        "expected_duration_months": dict(zip(REGIMES, expected_durations_months(P_).tolist())),
        "initial_regime_probs": dict(zip(REGIMES, np.asarray(init_probs).tolist())),
        "level_constraints": full.diagnostics["level_constraints"],
        "lw_shrinkage": full.diagnostics["lw_shrinkage"],
        "complete_case_months": full.diagnostics["complete_case_months"],
        "reference_params": ref,
        "bootstrap": {"replicates": len(reps), "block_lengths": block, "mean_block_length": L,
                      "fallback_frequency": {k: v / len(reps) for k, v in sorted(fb_counts.items())},
                      "replicates_with_any_fallback": sum(1 for f in fallback_log if f) / len(reps),
                      "param_percentiles": _param_percentiles(reps)},
    }
    report["house_state_space"] = {k: v for k, v in full.diagnostics["house_state_space"].items()}
    return {"ref": ref, "reps": reps, "report": report, "house_z": full.diagnostics["house_std_innovations"]}


def _param_percentiles(reps: list) -> dict:
    """5/50/95th percentiles of key parameters across replicates (estimation uncertainty)."""
    out = {}
    for blk in ("equity", "house", "short_rate", "term_spread", "mortgage_rate", "inflation"):
        for fld in ("mu", "sigma", "theta"):
            if fld in reps[0][blk]:
                A = np.array([r[blk][fld] for r in reps])
                out[f"{blk}.{fld}"] = {REGIMES[i]: np.percentile(A[:, i], [5, 50, 95]).round(5).tolist()
                                       for i in range(A.shape[1])}
    for blk, fld in (("house", "ar1"), ("short_rate", "half_life_months"), ("mortgage_rate", "half_life_months"),
                     ("term_spread", "half_life_months"), ("inflation", "half_life_months")):
        A = np.array([r[blk][fld] for r in reps])
        out[f"{blk}.{fld}"] = np.percentile(A, [5, 50, 95]).round(4).tolist()
    A = np.array([np.diag(r["transition_matrix"]) for r in reps])
    out["P_diag"] = {REGIMES[i]: np.percentile(A[:, i], [5, 50, 95]).round(3).tolist() for i in range(4)}
    for k in ("vrp_ratio", "skew"):
        A = np.array([r["implied_vol"][k] for r in reps])
        out[f"implied_vol.{k}"] = np.percentile(A, [5, 50, 95]).round(4).tolist()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261110)
    a = ap.parse_args()
    rep = main(a.replicates, a.seed)
    print(json.dumps({"sample": rep["sample"], "cross_check": {k: rep["regime_cross_check"][k] for k in ("agreement", "cohens_kappa")},
                      **{m: {k: v[k] for k in ("regime_months", "expected_duration_months", "initial_regime_probs")}
                         for m, v in rep["variants"].items()}}, indent=1))
