"""Glidepath optimisation: when to hand over equity -> protected equity -> short bonds -> cash.

Method: docs/METHODS.md section 10. Agreed with the user on 2026-11-11 before running.

Candidates ("relay" glidepaths)
  a = years before purchase at which equity hands over to protected equity  (8, 7, 6, 5, 4)
  b = ... protected equity hands over to short bonds                          (5, 4, 3, 2), b < a
  c = ... short bonds hand over to cash                                       (3, 2, 1, 0), c < b; 0 = no cash phase
  blend = share kept in the previous sleeve during each phase                 (0% hard switch, 25% soft)
  -> 53 orderings x 2 blends = 106 candidates. Today's agreed glidepath is evaluated as a reference.

Target (income test off, SaverProfile.apply_income_test = False)
  deposit target G = own_target_pct x house price at purchase + buying costs.
  Funded ratio F = pot / G (house-price growth is inside G, so F measures the deposit in house terms).

Measures per re-calibration (500 paths, common random numbers across all candidates and the bank account)
  growth = median F
  safety = CVaR_10%(F) = mean of the worst 10% of F (expected shortfall; a coherent risk measure,
           Artzner et al. 1999; Rockafellar & Uryasev 2000). Higher = safer.
  plus reported: P(F >= 1), P(pot > bank pot), P(money-weighted return >= house-price growth), P(all three),
  P(pot >= own funds needed with the income test), bank-account values of the same.

Selection rule (fixed before running)
  1. Safety bar: median over re-calibrations of the paired difference safety(candidate) - safety(bank) >= 0.
  2. Among candidates passing the bar under BOTH regime methods, pick the highest median growth (repo, primary).
  3. Decided on even-numbered re-calibrations; verified on odd-numbered (held out) and in the crisis replays.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from . import metrics as M
from .affordability_nl import FinancingCostTable, affordability_gap, load_policy
from .config import MarketParams
from .engine import RunConfig, _student_instalment, run, simulate_market_paths
from .glidepath import DEFAULT_TABLE, GlidepathSpec

A_YEARS = (8, 7, 6, 5, 4)
B_YEARS = (5, 4, 3, 2)
C_YEARS = (3, 2, 1, 0)
BLENDS = (0.0, 0.25)
CVAR_ALPHA = 0.10
METRICS = ("growth", "safety", "p_target", "p_beat_bank", "p_keep_up", "p_all3", "p_income_test", "pot_p50")
BANK = "bank"
TODAY = "today"


@dataclass(frozen=True)
class Relay:
    a: int
    b: int
    c: int
    blend: float

    @property
    def label(self) -> str:
        cash = f"cash {self.c}y" if self.c > 0 else "no cash"
        return f"PE {self.a}y | bonds {self.b}y | {cash} | blend {self.blend:.0%}"

    def table(self, horizon_years: int = 10) -> list[dict[str, Any]]:
        start = max(horizon_years, self.a + 1)
        e = lambda k: np.eye(4)[k]  # noqa: E731
        bl = self.blend
        rows = [(start, self.a, e(0)), (self.a, self.b, (1 - bl) * e(1) + bl * e(0))]
        if self.c > 0:
            rows += [(self.b, self.c, (1 - bl) * e(2) + bl * e(1)), (self.c, 0, (1 - bl) * e(3) + bl * e(2))]
        else:
            rows += [(self.b, 0, (1 - bl) * e(2) + bl * e(1))]
        return [{"from_years": float(f), "to_years": float(t), "weights": [float(x) for x in w]} for f, t, w in rows]

    def spec(self, horizon_years: int = 10) -> GlidepathSpec:
        return GlidepathSpec(kind="step", table=self.table(horizon_years), name=self.label)


def candidates() -> list[Relay]:
    return [Relay(a, b, c, bl) for bl in BLENDS for a in A_YEARS for b in B_YEARS if b < a
            for c in C_YEARS if c < b]


def cvar_low(x: np.ndarray, alpha: float = CVAR_ALPHA) -> float:
    """Mean of the worst ceil(alpha * n) values (lower tail expected shortfall of a 'higher is better' quantity)."""
    x = np.sort(np.asarray(x, float))
    k = max(1, int(np.ceil(alpha * x.size)))
    return float(x[:k].mean())


def outcome(V: np.ndarray, G: np.ndarray, B: np.ndarray, flows: np.ndarray, house_pa: np.ndarray,
            D_income: np.ndarray) -> dict[str, float]:
    F = V / G
    irr = (1.0 + M.irr_monthly(flows)) ** 12 - 1.0
    keep = np.where(np.isfinite(irr), irr >= house_pa, False)
    beat = V > B
    reach = F >= 1.0
    return {"growth": float(np.median(F)), "safety": cvar_low(F), "p_target": float(reach.mean()),
            "p_beat_bank": float(beat.mean()), "p_keep_up": float(keep.mean()),
            "p_all3": float((reach & beat & keep).mean()), "p_income_test": float((V >= D_income).mean()),
            "pot_p50": float(np.median(V))}


def _evaluate_replicate(args: tuple[RunConfig, tuple[GlidepathSpec, ...]]) -> np.ndarray:
    """(n_specs + 1, len(METRICS)): every glidepath, then the bank account (last row)."""
    cfg, specs = args
    m = simulate_market_paths(cfg)
    out = np.empty((len(specs) + 1, len(METRICS)))
    s, T = cfg.saver, m.n_months
    policy = load_policy(cfg.policy_year)
    table = FinancingCostTable.load(policy)
    D_inc = None
    for i, g in enumerate(specs):
        r = run(replace(cfg, glidepath=g), market_paths=m, summary=False)
        if D_inc is None:     # own funds needed WITH the income test (measure 2), same paths
            a = affordability_gap(r.price[:, -1], r.income[:, -1], m.mortgage_rate[:, -1],
                                  np.full(m.n_paths, _student_instalment(s, T)[T]), s.buyer.age_at_start + T / 12,
                                  m.cpi[:, -1], m.house_index[:, -1], s.buyer, policy, table, apply_income_test=True)
            D_inc = np.maximum(a.required_deposit,
                               s.own_target_pct * r.price[:, -1] + a.transfer_tax + a.nhg_premium + a.other_costs)
            house_pa = m.house_index[:, -1] ** (12.0 / T) - 1.0
            G = r.target_path[:, -1]
            Vb = r.benchmark.value[:, -1]
            fb = -r.benchmark.contributions.copy(); fb[:, -1] += Vb
            bank = outcome(Vb, G, Vb, fb, house_pa, D_inc)     # V > V is False: the bank never 'beats' itself
        V = r.portfolio.value[:, -1]
        f = -r.portfolio.contributions.copy(); f[:, -1] += V
        out[i] = [outcome(V, G, Vb, f, house_pa, D_inc)[k] for k in METRICS]
    out[-1] = [bank[k] for k in METRICS]
    return out


def evaluate(cfg: RunConfig, specs: Sequence[GlidepathSpec], params: Sequence[MarketParams], n_paths: int = 500,
             n_jobs: int | None = None) -> np.ndarray:
    """(R, n_specs + 1, len(METRICS)); the last spec row is the bank account."""
    jobs = [(replace(cfg, market=p, n_paths=n_paths), tuple(specs)) for p in params]
    n_jobs = n_jobs if n_jobs is not None else max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    if n_jobs == 1:
        return np.stack([_evaluate_replicate(j) for j in jobs])
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return np.stack(list(ex.map(_evaluate_replicate, jobs)))


def analyse(res: np.ndarray, labels: Sequence[str], idx: np.ndarray) -> dict[str, Any]:
    """Medians over the given re-calibrations, paired differences vs the bank (last row) and today (row 0)."""
    gi, si = METRICS.index("growth"), METRICS.index("safety")
    r = res[idx]
    med = np.median(r, axis=0)
    d_bank = r[:, :, si] - r[:, [-1], si]
    d_today_g = r[:, :, gi] - r[:, [0], gi]
    d_today_s = r[:, :, si] - r[:, [0], si]
    rows = []
    for j, lab in enumerate(labels):
        row = {"label": lab, **{k: float(med[j, i]) for i, k in enumerate(METRICS)},
               "safety_minus_bank_median": float(np.median(d_bank[:, j])),
               "share_safer_than_bank": float(np.mean(d_bank[:, j] >= 0)),
               "growth_minus_today_median": float(np.median(d_today_g[:, j])),
               "growth_minus_today_p05": float(np.percentile(d_today_g[:, j], 5)),
               "growth_minus_today_p95": float(np.percentile(d_today_g[:, j], 95)),
               "share_growth_above_today": float(np.mean(d_today_g[:, j] > 0)),
               "safety_minus_today_median": float(np.median(d_today_s[:, j])),
               "share_safety_above_today": float(np.mean(d_today_s[:, j] >= 0))}
        for i, k in enumerate(METRICS):
            row[f"{k}_p05"] = float(np.percentile(r[:, j, i], 5))
            row[f"{k}_p95"] = float(np.percentile(r[:, j, i], 95))
        row["passes_safety_bar"] = row["safety_minus_bank_median"] >= 0
        rows.append(row)
    return {"n": int(len(idx)), "rows": rows}


def choose(sel_by_method: dict[str, dict[str, Any]], primary: str = "repo") -> int | None:
    """Index of the highest-growth candidate (primary method) that passes the safety bar under every method.
    Candidate rows only: row 0 (today) and the last row (bank) are references, not candidates."""
    n = len(sel_by_method[primary]["rows"])
    ok = [j for j in range(1, n - 1) if all(m["rows"][j]["passes_safety_bar"] for m in sel_by_method.values())]
    if not ok:
        return None
    return max(ok, key=lambda j: sel_by_method[primary]["rows"][j]["growth"])
