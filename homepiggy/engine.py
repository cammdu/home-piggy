"""Top-level simulation: market -> sleeves -> portfolio -> affordability -> metrics.

Random numbers (docs/METHODS.md, section 8). A single integer seed feeds
numpy's SeedSequence, which spawns independent PCG64 streams for (a) the
initial regime, (b) regime transitions and (c) market innovations. Every
strategy run with the same seed and scenario therefore sees *identical* market
paths (common random numbers). Strategy comparisons are paired, which removes
market noise from the difference and narrows its confidence interval.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from . import metrics as M
from .affordability_nl import (
    AffordabilityResult, BuyerProfile, FinancingCostTable, affordability_gap, annuity_payment, load_policy,
)
from .affordability_de import FundTax, affordability_gap_de, load_policy_de
from .affordability_uk import affordability_gap_uk, lisa_charge, lisa_flows, load_policy_uk
from .config import STATE_VARS, MarketParams
from .glidepath import GlidepathSpec, static_weights, target_vol_lambda
from .market import MarketPaths, simulate_market
from .portfolio import PortfolioResult, RebalancePolicy, run_portfolio
from .regimes import RegimeSegment, simulate_regimes
from .scenarios import BASELINE, Scenario
from .sleeves import SLEEVES, ProductParams, ProtectedEquitySleeve, bond_returns, cash_returns, equity_returns
from .tax_nl import Box3Params


@dataclass(frozen=True)
class SaverProfile:
    """The saver's own inputs (scenario assumptions chosen by the user, not market data)."""

    horizon_years: int = 10
    initial_eur: float = 1000.0
    monthly_contribution_eur: float = 550.0
    gross_income_eur: float = 60000.0          # annual gross income at start (2026 euros)
    real_wage_growth: float = 0.01             # p.a. above CPI (career progression)
    house_price_eur: float = 400000.0          # target home's price today (2026 euros)
    student_debt_eur: float = 0.0
    student_rate: float = 0.0233               # DUO rate (user input; see docs/SOURCES.md)
    student_repay_start_month: int = 48        # months from start when DUO repayment begins
    student_term_months: int = 420             # SF35: 35-year repayment term
    own_target_pct: float = 0.10               # deposit goal: this share of the price at purchase + buying
                                               # costs; success = covering max(gap, own goal). 0 = costs only
    apply_income_test: bool = False            # True: the gap also includes the income-based borrowing limit
                                               # (Dutch "financieringslast"). Off by default (user decision
                                               # 2026-11-11): the income hurdle applies whatever the saver
                                               # does, so the fund is judged on the deposit + costs only.
    buyer: BuyerProfile = field(default_factory=BuyerProfile)

    def validate(self) -> "SaverProfile":
        if not (1 <= self.horizon_years <= 15):
            raise ValueError("horizon_years must be between 1 and 15.")
        for f in ("initial_eur", "monthly_contribution_eur", "gross_income_eur", "house_price_eur", "student_debt_eur"):
            if getattr(self, f) < 0:
                raise ValueError(f"{f} must be >= 0.")
        if self.house_price_eur <= 0:
            raise ValueError("house_price_eur must be > 0.")
        if not (0.0 <= self.own_target_pct <= 1.0):
            raise ValueError("own_target_pct must be between 0 and 1.")
        return self


# Default UK saver = the Dutch default converted to pounds (user decision 2026-11-23), so the two countries
# compare like for like. Rate: 0.8338 GBP per EUR, ECB reference rate, end of October 2026
# (EXR.M.GBP.EUR.SP00.E). Rounded to the page's slider steps: income 50,028 -> 50,000; price 333,520 ->
# 335,000; monthly 458.6 -> 460; initial 833.8 -> 834. Amount fields keep their *_eur names but hold pounds.
GBP_PER_EUR_DEFAULTS = 0.8338
UK_DEFAULT_SAVER = SaverProfile(initial_eur=834.0, monthly_contribution_eur=460.0, gross_income_eur=50_000.0,
                                house_price_eur=335_000.0)


def default_saver(country: str = "nl") -> SaverProfile:
    return UK_DEFAULT_SAVER if country == "uk" else SaverProfile()


@dataclass(frozen=True)
class RunConfig:
    market: MarketParams
    saver: SaverProfile = field(default_factory=SaverProfile)
    product: ProductParams = field(default_factory=ProductParams)
    glidepath: GlidepathSpec = field(default_factory=GlidepathSpec)
    rebalance: RebalancePolicy = field(default_factory=RebalancePolicy)
    scenario: Scenario = BASELINE
    n_paths: int = 5000
    seed: int = 20261110
    box3_enabled: bool = True
    box3_partners: bool = False
    other_box3_assets_eur: float = 0.0
    cash_sleeve_box3_category: str = "other"   # fund money-market sleeve counts as 'other assets'
    benchmark_savings_spread: float = 0.0050   # bank savings rate = short rate - this (benchmark)
    policy_year: int = 2026
    country: str = "nl"                        # "nl" (Dutch rules, box 3), "uk" (England, ISA wrappers) or "de"
    lisa: bool = True                          # UK only: Lifetime ISA bonus and charge (docs/METHODS.md section 12)


@dataclass
class Results:
    config: RunConfig
    market: MarketPaths
    portfolio: PortfolioResult
    benchmark: PortfolioResult
    afford: AffordabilityResult
    required_path: np.ndarray       # (P, T+1) own funds required if buying at month t (costs; + income gap if tested)
    target_path: np.ndarray         # (P, T+1) saver's goal: max(required, own_target_pct x price + costs)
    income: np.ndarray              # (P, T+1)
    price: np.ndarray               # (P, T+1)
    summary: dict[str, Any]
    lisa_charge: np.ndarray | None = None        # (P,) UK: Lifetime ISA withdrawal charge taken at purchase
    lisa_charge_bank: np.ndarray | None = None
    exit_tax: np.ndarray | None = None           # (P,) DE: tax on selling the fund at purchase


# --------------------------------------------------------------------------- random streams
def _streams(seed: int) -> tuple[np.random.Generator, ...]:
    ss = np.random.SeedSequence(seed)
    return tuple(np.random.Generator(np.random.PCG64(s)) for s in ss.spawn(3))


def simulate_market_paths(cfg: RunConfig) -> MarketPaths:
    p = cfg.market
    T = cfg.saver.horizon_years * 12
    g_init, g_trans, g_norm = _streams(cfg.seed)
    probs = p.initial_probs()
    init = np.searchsorted(np.cumsum(probs), g_init.random(cfg.n_paths), side="right")
    init = np.minimum(init, p.k - 1)
    uniforms = g_trans.random((cfg.n_paths, T))
    normals = g_norm.standard_normal((cfg.n_paths, T, len(STATE_VARS)))
    sc = cfg.scenario.resolve(T)
    regimes = simulate_regimes(p.P, cfg.n_paths, T, init, uniforms, sc.segments, p.regimes)
    return simulate_market(p, regimes, normals, sc.shocks)


# --------------------------------------------------------------------------- pilot covariance
def sleeve_covariances(market: MarketParams, product: ProductParams, n_paths: int = 2000,
                       months: int = 36, seed: int = 7) -> np.ndarray:
    """Annualised (K, 4, 4) covariance of monthly sleeve returns, conditional on each regime.

    Monte Carlo moment estimation: for each regime k the regime is forced for
    `months` months and the sample covariance of the simulated monthly sleeve
    returns is computed (months 13+ only, so the protected-equity ladder is in
    steady state). Used only by the target-volatility glidepath.
    """
    out = []
    g = np.random.default_rng(seed)
    for k, name in enumerate(market.regimes):
        regimes = np.full((n_paths, months + 1), k, dtype=np.int8)
        normals = g.standard_normal((n_paths, months, len(STATE_VARS)))
        m = simulate_market(market, regimes, normals)
        pe = ProtectedEquitySleeve(m, market, product)
        pe.set_value(np.ones(n_paths))
        v = [np.ones(n_paths)]
        for t in range(1, months + 1):
            v.append(pe.advance(t))
            pe.set_value(np.ones(n_paths))        # unit exposure each month -> returns = v - 1
        pe_ret = np.stack(v[1:], axis=1) - 1.0
        R = np.stack([equity_returns(m, product), pe_ret, bond_returns(m, product),
                      cash_returns(m, product.cash_spread)], axis=-1)[:, 12:, :]
        out.append(np.cov(R.reshape(-1, 4), rowvar=False) * 12.0)
    return np.stack(out)


# --------------------------------------------------------------------------- country rules
def load_rules(cfg: RunConfig) -> tuple[dict[str, Any], FinancingCostTable | None]:
    if cfg.country == "uk":
        return load_policy_uk(cfg.policy_year), None
    if cfg.country == "de":
        return load_policy_de(cfg.policy_year), None
    if cfg.country != "nl":
        raise ValueError(f"Unknown country '{cfg.country}'.")
    policy = load_policy(cfg.policy_year)
    return policy, FinancingCostTable.load(policy)


def afford(cfg: RunConfig, policy: dict[str, Any], table: FinancingCostTable | None, price: np.ndarray,
           income: np.ndarray, mortgage_rate: np.ndarray, instalment: np.ndarray, age: float, cpi: np.ndarray,
           house: np.ndarray, apply_income_test: bool) -> AffordabilityResult:
    """Affordability gap under the configured country's rules."""
    s = cfg.saver
    if cfg.country == "uk":
        return affordability_gap_uk(price, income, mortgage_rate, cpi, s.buyer, policy, apply_income_test)
    if cfg.country == "de":
        return affordability_gap_de(price, income, mortgage_rate, age, cpi, s.buyer, policy, apply_income_test)
    return affordability_gap(price, income, mortgage_rate, instalment, age, cpi, house, s.buyer, policy, table,
                             apply_income_test=apply_income_test)


# --------------------------------------------------------------------------- main run
def _income_index(m: MarketPaths, g: float) -> np.ndarray:
    t = np.arange(m.n_months + 1) / 12.0
    return m.cpi * (1.0 + g) ** t[None, :]


def _student_instalment(s: SaverProfile, T: int) -> np.ndarray:
    """Deterministic DUO instalment if buying at month t = 0..T (Art. 3a: actual debt, rate, remaining term)."""
    B = s.student_debt_eur
    out = np.zeros(T + 1)
    pay = None
    for t in range(T + 1):
        if t > 0:
            B *= 1.0 + s.student_rate / 12.0
            if t > s.student_repay_start_month:
                if pay is None:
                    pay = float(annuity_payment(B, s.student_rate, s.student_term_months))
                B = max(B - pay, 0.0)
        repaid = max(t - s.student_repay_start_month, 0)
        remaining = max(s.student_term_months - repaid, 1)
        out[t] = float(annuity_payment(B, s.student_rate, remaining)) if B > 0 else 0.0
    return out


def run(cfg: RunConfig, market_paths: MarketPaths | None = None,
        sleeve_cov: np.ndarray | None = None, summary: bool = True) -> Results:
    """Simulate one glidepath. summary=False skips summarise() (IRR etc.) for large studies."""
    s = cfg.saver.validate()
    prod = cfg.product.validate()
    T = s.horizon_years * 12
    m = market_paths if market_paths is not None else simulate_market_paths(cfg)
    if m.n_months != T:
        raise ValueError("market paths do not match the horizon.")
    P = m.n_paths
    policy, table = load_rules(cfg)
    uk = cfg.country == "uk"
    de = cfg.country == "de"

    # ---- saver cash flows, income, house price, student debt
    idx = _income_index(m, s.real_wage_growth)
    income = s.gross_income_eur * idx
    contributions = s.monthly_contribution_eur * idx
    contributions[:, 0] = 0.0
    price = s.house_price_eur * m.house_index
    instal = _student_instalment(s, T)

    # ---- required deposit if buying at each month (needed for lock-in and the final gap)
    required = np.empty((P, T + 1))
    target = np.empty((P, T + 1))
    for t in range(T + 1):
        a = afford(cfg, policy, table, price[:, t], income[:, t], m.mortgage_rate[:, t], np.full(P, instal[t]),
                   s.buyer.age_at_start + t / 12.0, m.cpi[:, t], m.house_index[:, t], s.apply_income_test)
        required[:, t] = a.required_deposit
        costs = a.transfer_tax + a.nhg_premium + a.other_costs
        target[:, t] = np.maximum(a.required_deposit, s.own_target_pct * price[:, t] + costs)
        if t == T:
            afford_T = a

    # ---- glidepath target function
    gp = cfg.glidepath
    W = static_weights(gp, T)
    locked_at = np.full(P, -1)
    if gp.kind == "target_vol":
        cov = sleeve_cov if sleeve_cov is not None else sleeve_covariances(cfg.market, prod)

        def target_fn(t: int, value: np.ndarray, reg: np.ndarray) -> np.ndarray:
            vt = gp.vol_start + (gp.vol_end - gp.vol_start) * (t / T)
            wb = np.repeat(W[t][None], P, axis=0)
            lam = target_vol_lambda(wb, cov[reg], vt)
            cash = np.zeros(len(SLEEVES)); cash[SLEEVES.index("cash")] = 1.0
            return lam[:, None] * wb + (1 - lam[:, None]) * cash
    elif gp.kind == "lock_in":
        lw = np.asarray(gp.lock_weights if gp.lock_weights is not None else W[-1], float)
        if lw.shape != (4,) or np.any(lw < 0) or not np.isclose(lw.sum(), 1):
            raise ValueError("lock_weights must be 4 non-negative weights summing to 1.")

        def target_fn(t: int, value: np.ndarray, reg: np.ndarray) -> np.ndarray:
            hit = (locked_at < 0) & (value >= gp.lock_multiple * target[:, t]) & (t > 0)
            locked_at[hit] = t
            return np.where((locked_at >= 0)[:, None], lw[None], W[t][None])
    else:
        def target_fn(t: int, value: np.ndarray, reg: np.ndarray) -> np.ndarray:
            return np.repeat(W[t][None], P, axis=0)

    ft_fund = ft_bank = None
    if uk:   # ISA wrappers: no tax on the pot; Lifetime ISA bonus invested alongside the saver's payments
        box3 = Box3Params(tax_rate=0.0, allowance=0.0, f_bank=0.0, f_other=0.0, enabled=False)
        lisa_pay, lisa_bonus = lisa_flows(contributions, s.buyer.age_at_start, policy, enabled=cfg.lisa)
    elif de:  # German fund taxation (Vorabpauschale, partial exemption, exit tax); savings interest taxed yearly
        box3 = Box3Params(tax_rate=0.0, allowance=0.0, f_bank=0.0, f_other=0.0, enabled=False)
        lisa_pay = lisa_bonus = np.zeros_like(contributions)
        ft_fund, ft_bank = FundTax.from_policy(policy, "fund"), FundTax.from_policy(policy, "interest")
    else:
        box3 = Box3Params.from_policy(policy, partners=cfg.box3_partners, enabled=cfg.box3_enabled,
                                      other_box3_assets=cfg.other_box3_assets_eur)
        lisa_pay = lisa_bonus = np.zeros_like(contributions)
    invested = contributions + lisa_bonus
    tracked = lisa_pay + lisa_bonus if uk else None
    rets = {"equity": equity_returns(m, prod), "short_bonds": bond_returns(m, prod),
            "cash": cash_returns(m, prod.cash_spread)}
    pe = ProtectedEquitySleeve(m, cfg.market, prod)
    cat = ("other", "other", "other", cfg.cash_sleeve_box3_category)
    port = run_portfolio(rets, pe, target_fn, m.regimes, s.initial_eur, invested, box3, cfg.rebalance, cat,
                         tracked=tracked, fund_tax=ft_fund)
    port.locked = locked_at.copy()

    # ---- benchmark: same contributions into a bank savings account (paired, same paths)
    cash_w = np.zeros(len(SLEEVES)); cash_w[SLEEVES.index("cash")] = 1.0
    bench_rets = {"equity": np.zeros_like(rets["cash"]), "short_bonds": np.zeros_like(rets["cash"]),
                  "cash": cash_returns(m, cfg.benchmark_savings_spread)}
    bench = run_portfolio(bench_rets, None, lambda t, v, r: np.repeat(cash_w[None], P, axis=0), m.regimes,
                          s.initial_eur, invested, box3, RebalancePolicy(transaction_cost=0.0),
                          ("other", "other", "other", "bank"), tracked=tracked, fund_tax=ft_bank)

    charge = charge_b = None
    if uk:   # the saver's own cash flows (for money-weighted returns) exclude the government bonus
        charge = lisa_charge(port.tracked_share * port.value[:, -1], price[:, -1], policy)
        charge_b = lisa_charge(bench.tracked_share * bench.value[:, -1], price[:, -1], policy)
        port.value[:, -1] -= charge
        bench.value[:, -1] -= charge_b
        port.contributions[:, 1:] = contributions[:, 1:]
        bench.contributions[:, 1:] = contributions[:, 1:]

    exit_tax = None
    if de:   # sale of all fund units at purchase (§ 19 InvStG with § 22 deemed-sale gains)
        st = port.fund_tax_state
        V = port.value[:, -1]
        exit_tax = np.minimum(ft_fund.tax_on(st["seg"] + (V - st["basis"] - st["vp_cum"]) * (1.0 - st["tf"])), V)
        port.value[:, -1] -= exit_tax
        port.tax_paid[:, -1] += exit_tax

    res = Results(cfg, m, port, bench, afford_T, required, target, income, price, {}, charge, charge_b, exit_tax)
    if summary:
        res.summary = summarise(res)
    return res


def compare(cfg: RunConfig, glidepaths: Sequence[GlidepathSpec]) -> dict[str, Results]:
    """Run several glidepaths on identical market paths (common random numbers)."""
    m = simulate_market_paths(cfg)
    cov = None
    if any(g.kind == "target_vol" for g in glidepaths):
        cov = sleeve_covariances(cfg.market, cfg.product)
    out = {}
    for g in glidepaths:
        if g.label in out:
            raise ValueError(f"Duplicate glidepath label '{g.label}'.")
        out[g.label] = run(replace(cfg, glidepath=g), market_paths=m, sleeve_cov=cov)
    return out


def paired_difference(a: Results, b: Results) -> dict[str, Any]:
    """A vs B on the same paths: P(A ends higher), mean difference with CI, success-rate difference."""
    va, vb = a.portfolio.value[:, -1], b.portfolio.value[:, -1]
    sa = va >= a.required_path[:, -1]
    sb = vb >= b.required_path[:, -1]
    d = sa.astype(float) - sb.astype(float)
    return {
        "p_a_higher_pot": M.prob(va > vb).as_dict(),
        "mean_pot_difference_eur": M.mean(va - vb).as_dict(),
        "success_rate_difference": M.mean(d).as_dict(),   # paired: CLT on per-path differences
    }


# --------------------------------------------------------------------------- summary
def summarise(r: Results) -> dict[str, Any]:
    m, port, bench = r.market, r.portfolio, r.benchmark
    T = m.n_months
    V = port.value[:, -1]
    Vc = bench.value[:, -1]
    D = r.required_path[:, -1]
    G = r.target_path[:, -1]
    cpi_T = m.cpi[:, -1]

    flows = -port.contributions.copy()
    flows[:, -1] += V
    irr_m = M.irr_monthly(flows)
    irr_a = (1.0 + irr_m) ** 12 - 1.0
    house_a = m.house_index[:, -1] ** (12.0 / T) - 1.0
    win = min(24, T)
    mdd_last = M.max_drawdown(port.nav[:, -win - 1:])
    shortfall = np.maximum(D - V, 0.0)
    fail = V < D
    qs = (0.05, 0.25, 0.5, 0.75, 0.95)

    def qdict(x: np.ndarray) -> dict[str, Any]:
        return {f"p{int(q*100):02d}": M.quantile(x, q).as_dict() for q in qs}

    return {
        "n_paths": int(V.size),
        "horizon_months": T,
        "glidepath": r.config.glidepath.label,
        "scenario": r.config.scenario.name,
        "p_reach_deposit": M.prob(V >= D).as_dict(),
        "p_outpace_cash": M.prob(V > Vc).as_dict(),
        "p_keep_up_with_housing": M.prob(irr_a >= house_a).as_dict(),
        "p_all_three": M.prob((V >= D) & (V > Vc) & (irr_a >= house_a)).as_dict(),
        "own_target_pct": r.config.saver.own_target_pct,
        "p_reach_own_target": M.prob(V >= G).as_dict(),
        "own_target": qdict(G),
        "surplus_vs_own_target": qdict(V - G),
        "pot_nominal": qdict(V),
        "pot_real_2026eur": qdict(V / cpi_T),
        "required_deposit": qdict(D),
        "surplus": qdict(V - D),
        "expected_shortfall_eur": M.mean(shortfall).as_dict(),
        "mean_shortfall_given_failure_eur": (M.mean(shortfall[fail]).as_dict() if fail.sum() > 1 else None),
        "money_weighted_return_pa": qdict(irr_a[np.isfinite(irr_a)]),
        "house_growth_pa": qdict(house_a),
        "max_drawdown_last_24m": qdict(mdd_last),
        "total_contributed_mean_eur": float(port.contributions.sum(axis=1).mean()),
        "box3_tax_mean_eur": float(port.tax_paid.sum(axis=1).mean()),
        "costs_mean_eur": float(port.costs_paid.sum(axis=1).mean()),
        "share_locked_in": float((port.locked >= 0).mean()),
        "regime_share": {name: float((m.regimes[:, 1:] == k).mean()) for k, name in enumerate(r.config.market.regimes)},
    }
