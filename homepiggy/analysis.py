"""Analysis layer used by the apps: option-C bands in parallel, plus path detail.

Everything here is a thin, testable wrapper around engine.compare() and
uncertainty.summarise_band(); no new statistics are introduced.

* ``band_study``: runs every glidepath on every bootstrap parameter set (common
  random numbers within and across replicates), in parallel processes, and
  returns band summaries per glidepath, paired success-rate differences against
  the first glidepath, and the index of the *median replicate*.
* ``path_detail``: re-runs one parameter set (the median replicate) with more
  paths to produce fan-chart quantiles, average allocations and the
  affordability decomposition at purchase. This is labelled in the app as "the
  median re-calibration", never pooled across replicates (option B was rejected).
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from .config import MarketParams
from .engine import RunConfig, compare
from .glidepath import GlidepathSpec
from .uncertainty import summarise_band

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def subset(params: Sequence[MarketParams], n: int) -> list[MarketParams]:
    """Deterministic, evenly spaced subset of n replicates (all if n >= len)."""
    if n >= len(params):
        return list(params)
    idx = np.unique(np.linspace(0, len(params) - 1, n).round().astype(int))
    return [params[i] for i in idx]


def _one_replicate(args: tuple[RunConfig, tuple[GlidepathSpec, ...]]) -> dict[str, Any]:
    cfg, gps = args
    res = compare(cfg, gps)
    out = {"summaries": {}, "success": {}}
    for g in gps:
        r = res[g.label]
        out["summaries"][g.label] = r.summary
        V = r.portfolio.value[:, -1]
        out["success"][g.label] = {"legal": (V >= r.required_path[:, -1]).astype(np.int8),
                                   "target": (V >= r.target_path[:, -1]).astype(np.int8)}
    return out


def band_study(cfg: RunConfig, glidepaths: Sequence[GlidepathSpec], params: Sequence[MarketParams],
               n_paths: int = 500, n_jobs: int | None = None) -> dict[str, Any]:
    if len({g.label for g in glidepaths}) != len(glidepaths):
        raise ValueError("Glidepath labels must be unique.")
    gps = tuple(glidepaths)
    jobs = [(replace(cfg, market=p, n_paths=n_paths), gps) for p in params]
    n_jobs = n_jobs if n_jobs is not None else max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    if n_jobs == 1:
        outs = [_one_replicate(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            outs = list(ex.map(_one_replicate, jobs, chunksize=max(1, len(jobs) // (4 * n_jobs))))

    bands = {g.label: summarise_band([o["summaries"][g.label] for o in outs], params) for g in gps}
    base = gps[0].label
    diffs = {}
    for g in gps[1:]:
        for kind in ("legal", "target"):
            d = np.array([o["success"][g.label][kind].mean() - o["success"][base][kind].mean() for o in outs])
            diffs.setdefault(g.label, {})[kind] = {
                "median": float(np.median(d)), "p05": float(np.percentile(d, 5)), "p95": float(np.percentile(d, 95)),
                "share_of_replicates_better": float(np.mean(d > 0)), "share_worse": float(np.mean(d < 0)),
            }
    key = np.array([o["summaries"][base]["p_reach_own_target"]["value"] for o in outs])
    median_rep = int(np.argsort(key)[len(key) // 2])
    return {"bands": bands, "paired_vs_first": diffs, "baseline": base, "median_replicate": median_rep,
            "n_replicates": len(outs), "n_paths": n_paths,
            "per_replicate": {g.label: [o["summaries"][g.label] for o in outs] for g in gps}}


def path_detail(cfg: RunConfig, glidepaths: Sequence[GlidepathSpec], params: MarketParams,
                n_paths: int = 2000) -> dict[str, Any]:
    """Fan-chart quantiles, mean allocations and affordability at purchase for one parameter set."""
    res = compare(replace(cfg, market=params, n_paths=n_paths), glidepaths)
    out: dict[str, Any] = {}
    for g in glidepaths:
        r = res[g.label]
        T = r.market.n_months
        months = np.arange(T + 1)
        q = lambda x: {f"p{int(k * 100):02d}": np.quantile(x, k, axis=0).tolist() for k in QUANTILES}  # noqa: E731
        a = r.afford
        out[g.label] = {
            "months": months.tolist(),
            "pot": q(r.portfolio.value),
            "pot_real": q(r.portfolio.value / r.market.cpi),
            "legal_min": q(r.required_path),
            "own_target": q(r.target_path),
            "cash_benchmark": q(r.benchmark.value),
            "weights_mean": r.portfolio.target.mean(axis=0).tolist(),
            "at_purchase": {
                "price": q(r.price[:, -1]), "max_loan": q(a.max_loan), "income_loan": q(a.income_loan),
                "transfer_tax": q(a.transfer_tax), "nhg": q(a.nhg_premium), "other_costs": q(a.other_costs),
                "legal_min": q(a.required_deposit), "own_target": q(r.target_path[:, -1]),
                "pot": q(r.portfolio.value[:, -1]), "income": q(r.income[:, -1]),
                "mortgage_rate": q(r.market.mortgage_rate[:, -1]), "test_rate": q(a.test_rate),
            },
            "summary": r.summary,
        }
    return out
