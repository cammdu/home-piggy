"""Uncertainty band (option C) for the default saver and the agreed glidepath, under both regime methods.

Run: PYTHONPATH=. .venv/bin/python -m calibration.band_default [--paths 1000]
Writes calibration/output/band_default.json (read by write_docs.py).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from homepiggy.engine import RunConfig, SaverProfile
from homepiggy.uncertainty import load_bootstrap, run_band

ROOT = Path(__file__).resolve().parents[1]


def main(n_paths: int = 1000) -> dict:
    saver = SaverProfile()
    out = {"n_paths": n_paths, "saver": {k: v for k, v in asdict(saver).items() if k != "buyer"}, "bands": {}}
    for method, f in (("repo_classifier", "nl_bootstrap.json"), ("hmm", "nl_bootstrap_hmm.json")):
        params, _ = load_bootstrap(ROOT / "params" / f)
        out["bands"][method] = run_band(RunConfig(market=params[0], saver=saver), params, n_paths=n_paths)
    (ROOT / "calibration" / "output" / "band_default.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=1000)
    main(ap.parse_args().paths)
