"""Parameter-uncertainty bands (option C).

The model is re-run once per bootstrap parameter set (params/nl_bootstrap.json).
Every set uses the same seed, so all replicates face the same random draws
(common random numbers) and differences between replicates reflect the
*parameters*, not simulation noise.

Reported per outcome:
  median across replicates   (the headline number)
  5th-95th percentile band   (plausible range given the data)
  mc_noise_ratio             median within-replicate 95% Monte Carlo half-width / band
                             half-width; must be small (< 0.25) for the band to be read
                             as parameter uncertainty
  drivers                    Spearman rank correlation between the outcome and summary
                             features of each replicate's parameters
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import spearmanr

from .config import MarketParams, market_params_from_dict
from .engine import RunConfig, compare, run
from .glidepath import GlidepathSpec

PROB_KEYS = ("p_reach_deposit", "p_outpace_cash", "p_keep_up_with_housing", "p_all_three")
QUANT_KEYS = (("pot_nominal", "p50"), ("required_deposit", "p50"), ("surplus", "p05"), ("surplus", "p50"),
              ("max_drawdown_last_24m", "p05"))


def load_bootstrap(path: str | Path) -> tuple[list[MarketParams], dict]:
    raw = json.loads(Path(path).read_text())
    prov = raw.get("provenance", {})
    if not prov.get("sources"):
        raise ValueError("Bootstrap file has no provenance.sources.")
    return [market_params_from_dict(r) for r in raw["replicates"]], prov


def occupancy(p: MarketParams, months: int = 120) -> np.ndarray:
    """Expected share of the next `months` spent in each regime, from today's regime probabilities.

    Always defined, including chains with an absorbing regime (where no unique stationary
    distribution exists), and it is the horizon that matters for a 10-year saver.
    """
    q = p.initial_probs()
    acc = np.zeros_like(q)
    for _ in range(months):
        q = q @ p.P
        acc += q
    return acc / months


def features(p: MarketParams) -> dict[str, float]:
    w = occupancy(p)
    return {
        "equity_mu (10y occupancy avg)": float(w @ p.equity.mu),
        "house_mu (10y occupancy avg)": float(w @ p.house.mu),
        "house_ar1": float(p.house_ar1),
        "mortgage_theta (10y occupancy avg)": float(w @ p.levels["mortgage_rate"].theta),
        "short_rate_theta (10y occupancy avg)": float(w @ p.levels["short_rate"].theta),
        "share of next 10y in Recession": float(w[list(p.regimes).index("Recession")]) if "Recession" in p.regimes else np.nan,
        "vrp_ratio": float(p.implied_vol.vrp_ratio),
    }


@dataclass
class Band:
    median: float
    lo: float
    hi: float
    mc_noise_ratio: float
    values: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {"median": self.median, "p05": self.lo, "p95": self.hi, "mc_noise_ratio": self.mc_noise_ratio}


def _band(vals: np.ndarray, halfwidths: np.ndarray) -> Band:
    lo, med, hi = np.percentile(vals, [5, 50, 95])
    bw = max((hi - lo) / 2, 1e-12)
    return Band(float(med), float(lo), float(hi), float(np.median(halfwidths) / bw), vals)


def summarise_band(summaries: Sequence[dict], params: Sequence[MarketParams]) -> dict[str, Any]:
    out: dict[str, Any] = {"replicates": len(summaries)}
    feats = [features(p) for p in params]
    fnames = list(feats[0])
    F = np.array([[f[n] for n in fnames] for f in feats])
    for k in PROB_KEYS:
        v = np.array([s[k]["value"] for s in summaries])
        hw = np.array([(s[k]["hi"] - s[k]["lo"]) / 2 for s in summaries])
        b = _band(v, hw)
        d = {}
        if np.ptp(v) > 0:
            for j, n in enumerate(fnames):
                if np.ptp(F[:, j]) > 0:
                    d[n] = float(spearmanr(F[:, j], v).statistic)
        out[k] = {**b.as_dict(), "drivers_spearman": dict(sorted(d.items(), key=lambda kv: -abs(kv[1])))}
    for k, q in QUANT_KEYS:
        v = np.array([s[k][q]["value"] for s in summaries])
        hw = np.array([(s[k][q]["hi"] - s[k][q]["lo"]) / 2 for s in summaries])
        out[f"{k}.{q}"] = _band(v, hw).as_dict()
    return out


def run_band(cfg: RunConfig, params: Sequence[MarketParams], n_paths: int = 1000) -> dict[str, Any]:
    """Run one glidepath under every parameter set; return the band summary."""
    sums = [run(replace(cfg, market=p, n_paths=n_paths)).summary for p in params]
    return summarise_band(sums, params)


def compare_band(cfg: RunConfig, glidepaths: Sequence[GlidepathSpec], params: Sequence[MarketParams],
                 n_paths: int = 1000) -> dict[str, Any]:
    """Band summary per glidepath, plus a band for paired success-rate differences vs the first glidepath."""
    per: dict[str, list] = {g.label: [] for g in glidepaths}
    diffs: dict[str, list] = {g.label: [] for g in glidepaths[1:]}
    base = glidepaths[0].label
    for p in params:
        res = compare(replace(cfg, market=p, n_paths=n_paths), glidepaths)
        for g in glidepaths:
            per[g.label].append(res[g.label].summary)
        a = res[base]
        sa = a.portfolio.value[:, -1] >= a.required_path[:, -1]
        for g in glidepaths[1:]:
            r = res[g.label]
            sb = r.portfolio.value[:, -1] >= r.required_path[:, -1]
            diffs[g.label].append(float(sb.mean() - sa.mean()))
    out = {g: summarise_band(s, params) for g, s in per.items()}
    out["_success_rate_difference_vs_" + base] = {
        g: {"median": float(np.median(v)), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
            "share_of_replicates_better": float(np.mean(np.array(v) > 0))}
        for g, v in diffs.items()
    }
    return out
