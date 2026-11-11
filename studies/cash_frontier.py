"""Run the cash-conversion frontier study (homepiggy/frontier.py) for both regime methods.

Usage:  PYTHONPATH=. python studies/cash_frontier.py [--paths 500] [--replicates 200] [--method repo|hmm|both]

Outputs (studies/output/):
  cash_frontier_<method>.npz   raw array (replicate x candidate x metric), for re-analysis
  cash_frontier_<method>.json  analysis (frontier on selection half, verification on test half)
Default saver (engine.SaverProfile): EUR 60k income, EUR 400k house, EUR 550/month, 10y, single.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import date
from pathlib import Path

import numpy as np

from homepiggy.analysis import subset
from homepiggy.engine import RunConfig, SaverProfile
from homepiggy.frontier import CASH_LEVELS, METRICS, STARTS, analyse, evaluate, grid
from homepiggy.uncertainty import load_bootstrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "studies" / "output"
FILES = {"repo": ROOT / "params" / "nl_bootstrap.json", "hmm": ROOT / "params" / "nl_bootstrap_hmm.json"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=500)
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--method", choices=("repo", "hmm", "both"), default="both")
    ap.add_argument("--seed", type=int, default=20261110)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cands = grid()
    for method in (("repo", "hmm") if a.method == "both" else (a.method,)):
        path = FILES[method]
        params, prov = load_bootstrap(path)
        params = subset(params, a.replicates)
        cfg = RunConfig(market=params[0], saver=SaverProfile(), seed=a.seed)
        t0 = time.time()
        res = evaluate(cfg, cands, params, n_paths=a.paths)
        secs = time.time() - t0
        np.savez_compressed(OUT / f"cash_frontier_{method}.npz", res=res, metrics=np.array(METRICS))
        out = {k: analyse(res, cands, safety=k) for k in ("bad_case", "bad_vs_bank")}
        out["setup"] = {
            "method": method, "params_file": str(path.relative_to(ROOT)),
            "params_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_paths": a.paths, "seed": a.seed, "starts_years": list(STARTS), "cash_levels": list(CASH_LEVELS),
            "saver": {k: v for k, v in vars(SaverProfile()).items() if k != "buyer"},
            "run_date": date.today().isoformat(), "runtime_s": round(secs, 1),
        }
        (OUT / f"cash_frontier_{method}.json").write_text(json.dumps(out, indent=1))
        for k in ("bad_case", "bad_vs_bank"):
            o = out[k]
            print(f"{method}/{k}: {len(cands)} candidates x {len(params)} re-calibrations in {secs/60:.1f} min; "
                  f"frontier sel/test {o['n_frontier_selection']}/{o['n_frontier_test']}, "
                  f"Jaccard {o['frontier_overlap_jaccard']:.2f}, baseline on frontier {o['baseline_on_frontier']}",
                  flush=True)


if __name__ == "__main__":
    main()
