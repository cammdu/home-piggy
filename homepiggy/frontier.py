"""Cash-conversion frontier study (growth vs safety). Method: docs/METHODS.md section 10.

Question: when should the fund start converting to cash, and how much cash should it hold
in the two late phases? Answered with a grid search, not an optimiser, so every candidate
and its evidence is visible.

Candidate glidepaths (all other settings as in the agreed glidepath, DEFAULT_TABLE):
  * ``start``  - years before purchase at which the cash-conversion phase begins (6, 5, 4, 3).
    Today: 4. The two growth phases keep today's weights (10-7y; 7y-start).
  * ``c_mid``  - cash share from ``start`` until 2 years before purchase (0, 10, ..., 100%).
  * ``c_late`` - cash share in the last 2 years (0, 10, ..., 100%), with c_late >= c_mid
    (cash never falls as purchase approaches: that is what "converting to cash" means).
  * The non-cash remainder keeps today's composition within that phase (proportional
    scaling): mid phase 40% protected equity / 60% short bonds, late phase 100% short bonds.
    Today's glidepath (start 4, c_mid 50%, c_late 80%) is therefore one grid point.

Two outcome measures per candidate, computed on each bootstrap re-calibration (option C)
with common random numbers across candidates:
  * growth = P(pot >= own target at purchase)  (higher is better);
  * safety = bad case = 5th percentile of (pot - own target) in EUR  (higher is better;
    negative = euros short of target in the worst 1-in-20 outcome);
  * safety (alternative) = 5th percentile of (pot - bank-savings pot) in EUR: the worst
    1-in-20 regret versus simply saving the same contributions in a bank account. Added
    because the target's own tail (house prices, mortgage rates) dominates the first
    measure and no allocation can move it much; this one isolates what the fund controls.

Frontier: a candidate is *dominated* if another is at least as good on both medians
(across re-calibrations) and strictly better on one. The non-dominated set is the frontier.
Selection uses the even-numbered re-calibrations; everything is re-checked on the
odd-numbered half (held out) so that frontier membership is not an artefact of noise.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from .config import MarketParams
from .engine import RunConfig, run, simulate_market_paths
from .glidepath import DEFAULT_TABLE, GlidepathSpec

STARTS = (6.0, 5.0, 4.0, 3.0)
CASH_LEVELS = tuple(round(0.1 * i, 1) for i in range(11))
LATE_FROM = 2.0
BASELINE = (4.0, 0.5, 0.8)
MID_NONCASH = np.array([0.0, 0.4, 0.6, 0.0])    # today's 4-2y phase without cash: 20/30 -> 40/60
LATE_NONCASH = np.array([0.0, 0.0, 1.0, 0.0])   # today's 2-0y phase without cash: all short bonds
METRICS = ("p_target", "bad_case", "p_legal", "median_surplus", "bad_vs_bank")
SAFETY = {"bad_case": "pot - own target", "bad_vs_bank": "pot - bank-savings pot"}


@dataclass(frozen=True)
class Candidate:
    start: float
    c_mid: float
    c_late: float

    @property
    def label(self) -> str:
        return f"start {self.start:g}y | cash {self.c_mid:.0%} → {self.c_late:.0%}"

    def table(self) -> list[dict[str, Any]]:
        g1, g2 = DEFAULT_TABLE[0], DEFAULT_TABLE[1]
        mid = MID_NONCASH * (1 - self.c_mid) + np.array([0, 0, 0, self.c_mid])
        late = LATE_NONCASH * (1 - self.c_late) + np.array([0, 0, 0, self.c_late])
        rows = [dict(g1)]
        if self.start < g2["from_years"]:
            rows.append({"from_years": g2["from_years"], "to_years": self.start, "weights": list(g2["weights"])})
        else:   # start == 7 would remove the second growth phase; not in the grid
            raise ValueError("start must be below 7 years.")
        rows.append({"from_years": self.start, "to_years": LATE_FROM, "weights": mid.round(12).tolist()})
        rows.append({"from_years": LATE_FROM, "to_years": 0.0, "weights": late.round(12).tolist()})
        return rows

    def spec(self) -> GlidepathSpec:
        return GlidepathSpec(kind="step", table=self.table(), name=self.label)


def grid(starts: Sequence[float] = STARTS, levels: Sequence[float] = CASH_LEVELS) -> list[Candidate]:
    out = [Candidate(s, m, l) for s in starts for m in levels for l in levels if l >= m - 1e-12]
    if not any(c == Candidate(*BASELINE) for c in out):
        raise ValueError("The grid must contain today's glidepath as reference.")
    return out


def outcome_metrics(V: np.ndarray, G: np.ndarray, D: np.ndarray, B: np.ndarray | None = None) -> dict[str, float]:
    """V pot, G own target, D legal minimum, B bank-savings pot (same contributions), all at purchase."""
    s = V - G
    out = {"p_target": float(np.mean(V >= G)), "bad_case": float(np.quantile(s, 0.05)),
           "p_legal": float(np.mean(V >= D)), "median_surplus": float(np.median(s))}
    out["bad_vs_bank"] = float(np.quantile(V - B, 0.05)) if B is not None else float("nan")
    return out


def _evaluate_replicate(args: tuple[RunConfig, tuple[Candidate, ...]]) -> np.ndarray:
    cfg, cands = args
    m = simulate_market_paths(cfg)          # common random numbers across candidates
    out = np.empty((len(cands), len(METRICS)))
    for i, c in enumerate(cands):
        r = run(replace(cfg, glidepath=c.spec()), market_paths=m, summary=False)
        met = outcome_metrics(r.portfolio.value[:, -1], r.target_path[:, -1], r.required_path[:, -1],
                              r.benchmark.value[:, -1])
        out[i] = [met[k] for k in METRICS]
    return out


def evaluate(cfg: RunConfig, cands: Sequence[Candidate], params: Sequence[MarketParams],
             n_paths: int = 500, n_jobs: int | None = None) -> np.ndarray:
    """Array (R, C, len(METRICS)): every candidate on every re-calibration."""
    jobs = [(replace(cfg, market=p, n_paths=n_paths), tuple(cands)) for p in params]
    n_jobs = n_jobs if n_jobs is not None else max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    if n_jobs == 1:
        return np.stack([_evaluate_replicate(j) for j in jobs])
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return np.stack(list(ex.map(_evaluate_replicate, jobs)))


def pareto_mask(growth: np.ndarray, safety: np.ndarray) -> np.ndarray:
    """True where no other point is >= on both and > on at least one."""
    g, s = np.asarray(growth, float), np.asarray(safety, float)
    ge = (g[None, :] >= g[:, None]) & (s[None, :] >= s[:, None])
    gt = (g[None, :] > g[:, None]) | (s[None, :] > s[:, None])
    return ~np.any(ge & gt, axis=1)


def analyse(res: np.ndarray, cands: Sequence[Candidate], safety: str = "bad_case") -> dict[str, Any]:
    """Selection on even re-calibrations, verification on odd ones. `safety` is a key of SAFETY."""
    R = res.shape[0]
    sel, test = np.arange(0, R, 2), np.arange(1, R, 2)
    gi, si = METRICS.index("p_target"), METRICS.index(safety)
    b = [i for i, c in enumerate(cands) if c == Candidate(*BASELINE)][0]

    def half(idx: np.ndarray) -> dict[str, Any]:
        med = np.median(res[idx], axis=0)                  # (C, metrics)
        front = pareto_mask(med[:, gi], med[:, si])
        d = res[idx] - res[idx][:, [b], :]                 # paired vs today's glidepath
        better_both = np.mean((d[:, :, gi] > 0) & (d[:, :, si] > 0), axis=0)
        return {"median": med, "frontier": front,
                "dg_median": np.median(d[:, :, gi], axis=0), "ds_median": np.median(d[:, :, si], axis=0),
                "dg_p05": np.percentile(d[:, :, gi], 5, axis=0), "dg_p95": np.percentile(d[:, :, gi], 95, axis=0),
                "ds_p05": np.percentile(d[:, :, si], 5, axis=0), "ds_p95": np.percentile(d[:, :, si], 95, axis=0),
                "share_better_growth": np.mean(d[:, :, gi] > 0, axis=0),
                "share_better_safety": np.mean(d[:, :, si] > 0, axis=0),
                "share_better_both": better_both}

    S, T = half(sel), half(test)
    fs, ft = set(np.flatnonzero(S["frontier"])), set(np.flatnonzero(T["frontier"]))
    rows = []
    for i, c in enumerate(cands):
        row = {"start": c.start, "c_mid": c.c_mid, "c_late": c.c_late, "label": c.label,
               "is_baseline": i == b, "frontier_selection": bool(S["frontier"][i]),
               "frontier_test": bool(T["frontier"][i])}
        for k, j in (("growth", gi), ("bad_case", si)):
            row[f"{k}_sel"] = float(S["median"][i, j])
            row[f"{k}_test"] = float(T["median"][i, j])
        row["p_legal_test"] = float(T["median"][i, METRICS.index("p_legal")])
        row["median_surplus_test"] = float(T["median"][i, METRICS.index("median_surplus")])
        for k in ("dg_median", "dg_p05", "dg_p95", "ds_median", "ds_p05", "ds_p95",
                  "share_better_growth", "share_better_safety", "share_better_both"):
            row[f"{k}_test"] = float(T[k][i])
        rows.append(row)
    dominating = [i for i in fs if T["share_better_both"][i] > 0.5]
    return {
        "safety_metric": safety, "safety_definition": f"5th percentile of ({SAFETY[safety]}) at purchase, EUR",
        "n_replicates": R, "n_selection": len(sel), "n_test": len(test), "n_candidates": len(cands),
        "baseline_index": b,
        "baseline_on_frontier": {"selection": bool(S["frontier"][b]), "test": bool(T["frontier"][b])},
        "frontier_overlap_jaccard": len(fs & ft) / max(1, len(fs | ft)),
        "n_frontier_selection": len(fs), "n_frontier_test": len(ft),
        "frontier_beating_baseline_on_both_in_majority_of_test": sorted(int(i) for i in dominating),
        "rows": rows,
    }
