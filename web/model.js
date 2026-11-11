// What the pitch page computes, on top of engine.js. Same statistics as the Python study
// (homepiggy/glidepath_opt.py) and the Streamlit app: option-C bands across bootstrap re-calibrations,
// then path detail for the median re-calibration. No new model.
//
// Feasibility measure 1 (default): pot >= deposit target (share of price + buying costs, income test off)
//   AND pot > savings account AND money-weighted return >= house-price growth.
// Feasibility measure 2: pot >= own funds needed when the Dutch income limit also applies.
import { runMany, quantile, cvarLow, affordability, bsPut, DEFAULT_TABLE, PRODUCT, historicalScenario } from "./engine.js";

export const ALL_EQUITY = [{ from_years: 15, to_years: 0, weights: [1, 0, 0, 0] }];
export const SEED = 20261110;
/** Run sizes used by the page and by the precompute step (identical, so precomputed = live). */
export const CONFIG = { nReps: 40, bandPaths: 400, detailPaths: 1000, stressPaths: 600 };

export function strategies(optimisedTable) {
  return [
    { key: "opt", label: "Home Piggy (optimised)", table: optimisedTable },
    { key: "today", label: "Original glidepath", table: DEFAULT_TABLE },
    { key: "equity", label: "100% global equity", table: ALL_EQUITY },
  ];
}
export const KEYS = ["opt", "today", "equity", "bank"];
export const STATS = ["p_all3", "p_target", "p_beat_bank", "p_keep_up", "p_income", "growth", "safety", "pot_p50", "pot_p05"];

/** Evenly spaced replicate indices (analysis.subset). */
export function subset(n, k) {
  if (k >= n) return [...Array(n).keys()];
  const out = new Set();
  for (let i = 0; i < k; i++) out.add(Math.round((i * (n - 1)) / (k - 1)));
  return [...out];
}

function stats(V, irr, r, isBank) {
  const n = V.length, F = new Float64Array(n);
  let reach = 0, beat = 0, keep = 0, all3 = 0, inc = 0;
  for (let i = 0; i < n; i++) {
    F[i] = V[i] / r.G[i];
    const a = V[i] >= r.G[i], b = !isBank && V[i] > r.B[i], k = Number.isFinite(irr[i]) && irr[i] >= r.housePa[i];
    reach += a; beat += b; keep += k; all3 += a && b && k; inc += V[i] >= r.Ginc[i];
  }
  return { p_target: reach / n, p_beat_bank: beat / n, p_keep_up: keep / n, p_all3: all3 / n, p_income: inc / n,
           growth: quantile(F, 0.5), safety: cvarLow(F, 0.10), pot_p50: quantile(V, 0.5), pot_p05: quantile(V, 0.05) };
}

/** One re-calibration: statistics for every strategy and the savings account, on the same paths. */
export function replicateSummary(params, policy, saver, nPaths, optTable, scenario = null) {
  const S = strategies(optTable);
  const r = runMany(params, policy, saver, { nPaths, seed: SEED, tables: S.map(s => s.table), scenario });
  const out = { target_p50: quantile(r.G, 0.5), income_target_p50: quantile(r.Ginc, 0.5) };
  S.forEach((s, k) => { out[s.key] = stats(r.strategies[k].V, r.strategies[k].irr, r, false); });
  out.bank = stats(r.B, r.bankIrr, r, true);
  return out;
}

/** Band across re-calibrations (uncertainty.summarise_band): median and 5-95% of each statistic. */
export function band(summaries) {
  const pick = (f) => { const v = summaries.map(f).filter(Number.isFinite);
    return { median: quantile(v, 0.5), p05: quantile(v, 0.05), p95: quantile(v, 0.95) }; };
  const out = { n: summaries.length, target_p50: pick(s => s.target_p50), income_target_p50: pick(s => s.income_target_p50) };
  for (const k of KEYS) { out[k] = {}; for (const m of STATS) out[k][m] = pick(s => s[k][m]); }
  const share = (f) => summaries.filter(f).length / summaries.length;
  out.paired = {
    opt_safer_than_bank: share(s => s.opt.safety >= s.bank.safety),
    opt_growth_above_today: share(s => s.opt.growth > s.today.growth),
    opt_safer_than_equity: share(s => s.opt.safety >= s.equity.safety),
    opt_all3_above_today: share(s => s.opt.p_all3 > s.today.p_all3),
  };
  const order = summaries.map((s, i) => [s.opt.p_all3, i]).sort((a, b) => a[0] - b[0]);
  out.median_pos = order[Math.floor(order.length / 2)][1];
  return out;
}

const Q = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Fan-chart quantiles and a typical purchase for one parameter set (analysis.path_detail). */
export function detail(params, policy, saver, nPaths, optTable) {
  const S = strategies(optTable);
  const r = runMany(params, policy, saver, { nPaths, seed: SEED, tables: S.map(s => s.table), keepPaths: true });
  const T = r.T, T1 = T + 1, months = [];
  for (let t = 0; t <= T; t += 3) months.push(t);
  if (months[months.length - 1] !== T) months.push(T);
  const col = (arr, t) => { const c = new Float64Array(r.nPaths); for (let i = 0; i < r.nPaths; i++) c[i] = arr[i * T1 + t]; return c; };
  const fan = months.map(t => {
    const v = col(r.strategies[0].Vpath, t), row = { t, year: t / 12 };
    Q.forEach(q => { row[`v${Math.round(q * 100)}`] = quantile(v, q); });
    row.today50 = quantile(col(r.strategies[1].Vpath, t), 0.5);
    row.e50 = quantile(col(r.strategies[2].Vpath, t), 0.5);
    row.b50 = quantile(col(r.Bpath, t), 0.5);
    row.g50 = quantile(r.Gpath[t], 0.5);
    return row;
  });
  // typical purchase: the path whose deposit target is the median target (one consistent path)
  const G = r.G, gm = quantile(G, 0.5);
  let best = 0; for (let i = 1; i < G.length; i++) if (Math.abs(G[i] - gm) < Math.abs(G[best] - gm)) best = i;
  const o = best * T1 + T, s = r.saver, m = r.market;
  const price = s.house_price_eur * m.houseIndex[o];
  const a = affordability(price, r.income[o], m.mortgageRate[o], r.instal[T], s.age_at_start + T / 12, m.cpi[o], m.houseIndex[o],
                          { ...s, apply_income_test: true }, policy);
  const typical = { price, income: r.income[o], mortgage_rate: m.mortgageRate[o], test_rate: a.test, flp: a.flp,
    income_loan: a.incomeLoan, transfer_tax: a.tt, nhg: a.nhg, other: a.other, deposit_target: G[best],
    deposit_pct: s.own_target_pct, income_gap: a.gap, income_target: r.Ginc[best], age: s.age_at_start + T / 12,
    house_growth_pa: r.housePa[best] };
  return { T, fan, weights: S.map((x, k) => ({ key: x.key, W: r.strategies[k].W })), typical };
}

/** Crisis replay ending at the purchase date, on one parameter set, vs the same paths without it. */
export function stress(params, policy, saver, event, nPaths, optTable) {
  const T = (saver.horizon_years || 10) * 12;
  const base = replicateSummary(params, policy, saver, nPaths, optTable, null);
  const hit = replicateSummary(params, policy, saver, nPaths, optTable, historicalScenario(event, T, 0));
  return { base, hit, event: event.name, window: event.window, duration: event.duration_months };
}

/** 12-month payoff of one buffer tranche vs the index (options.py, sleeves._strike), per unit invested. */
export function bufferPayoff(params, regimeIdx) {
  const iv = params.implied_vol, r = params.levels.short_rate.initial;
  const atm = iv.vrp_ratio * params.equity.sigma[regimeIdx];
  const k1 = PRODUCT.buffer_upper_strike, k2 = PRODUCT.buffer_lower_strike;
  const vol = (k) => Math.max(iv.min_vol, atm + iv.skew * (-Math.log(k)));
  const p1 = bsPut(1, k1, 1, r, vol(k1)), p2 = bsPut(1, k2, 1, r, vol(k2));
  const cost = (p1 - p2) + PRODUCT.option_cost;                 // per unit of index, paid up front
  const pts = [];
  for (let i = -40; i <= 40; i++) {
    const x = i / 100, ST = 1 + x, payoff = ST + Math.max(k1 - ST, 0) - Math.max(k2 - ST, 0);
    pts.push({ index: x, buffer: payoff / (1 + cost) - 1 });
  }
  return { cost, spread: p1 - p2, atm, rate: r, pts };
}
