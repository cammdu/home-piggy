"""Run the glidepath optimisation (homepiggy/glidepath_opt.py) with the pre-agreed selection rule.

Usage: PYTHONPATH=. python studies/optimize_glidepath.py [--paths 500] [--replicates 200] [--crisis-reps 40]
Output: studies/output/glidepath_opt.json (+ raw .npz per method).
Default saver (engine.SaverProfile): EUR 60k income, EUR 400k house, EUR 550/month, 10y, deposit 10% + costs,
income test off.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np

from homepiggy.analysis import subset
from homepiggy.engine import RunConfig, SaverProfile
from homepiggy.glidepath import DEFAULT_TABLE, GlidepathSpec
from homepiggy.glidepath_opt import (BANK, METRICS, TODAY, analyse, candidates, choose, evaluate)
from homepiggy.scenarios import historical_scenario
from homepiggy.uncertainty import load_bootstrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "studies" / "output"
FILES = {"repo": ROOT / "params" / "nl_bootstrap.json", "hmm": ROOT / "params" / "nl_bootstrap_hmm.json"}
EVENT_KEY = {"repo": "repo_classifier", "hmm": "hmm"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=500)
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--crisis-reps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261110)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cands = candidates()
    specs = [GlidepathSpec(kind="step", table=DEFAULT_TABLE, name=TODAY)] + [c.spec() for c in cands]
    labels = [s.label for s in specs] + [BANK]
    saver = SaverProfile()
    events = json.loads((ROOT / "params" / "nl_scenarios.json").read_text())["events"]

    sel, test, raw, params_by = {}, {}, {}, {}
    for method, path in FILES.items():
        params = subset(load_bootstrap(path)[0], a.replicates)
        params_by[method] = params
        cfg = RunConfig(market=params[0], saver=saver, seed=a.seed)
        t0 = time.time()
        res = evaluate(cfg, specs, params, n_paths=a.paths)
        np.savez_compressed(OUT / f"glidepath_opt_{method}.npz", res=res, metrics=np.array(METRICS),
                            labels=np.array(labels))
        R = res.shape[0]
        sel[method] = analyse(res, labels, np.arange(0, R, 2))
        test[method] = analyse(res, labels, np.arange(1, R, 2))
        raw[method] = {"runtime_min": round((time.time() - t0) / 60, 1), "n_replicates": R,
                       "params_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"{method}: {len(specs)} glidepaths x {R} re-calibrations in {raw[method]['runtime_min']} min", flush=True)

    win = choose(sel)
    result = {"setup": {"saver": {k: v for k, v in vars(saver).items() if k != "buyer"}, "n_paths": a.paths,
                        "seed": a.seed, "run_date": date.today().isoformat(), "methods": raw,
                        "rule": "max median growth (repo, selection half) s.t. median paired safety - bank >= 0 "
                                "under both methods (selection half); verified on held-out half and crisis replays"},
              "labels": labels, "candidates": [vars(c) for c in cands], "selection": sel, "heldout": test,
              "winner_index": win}
    if win is None:
        print("No candidate passes the safety bar under both methods.", flush=True)
    else:
        c = cands[win - 1]
        result["winner"] = {"label": labels[win], **vars(c), "table": c.table()}
        print("winner:", labels[win], flush=True)
        # crisis replays ending at the purchase date: today vs winner vs bank, same rule
        crisis = {}
        for method in FILES:
            ps = subset(params_by[method], a.crisis_reps)
            crisis[method] = {}
            for ev in events[EVENT_KEY[method]]:
                sc = historical_scenario(ev, saver.horizon_years * 12, 0)
                cfg = RunConfig(market=ps[0], saver=saver, seed=a.seed, scenario=sc)
                res = evaluate(cfg, [specs[0], specs[win]], ps, n_paths=a.paths)
                an = analyse(res, [TODAY, labels[win], BANK], np.arange(res.shape[0]))
                crisis[method][ev["name"]] = {"window": ev["window"], "rows": an["rows"]}
                w = an["rows"][1]
                print(f"  {method} {ev['name']}: winner safety-bank {w['safety_minus_bank_median']:+.3f}, "
                      f"growth {w['growth']:.3f} (today {an['rows'][0]['growth']:.3f})", flush=True)
        result["crisis"] = crisis
    (OUT / "glidepath_opt.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
