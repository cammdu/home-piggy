// home-piggy browser engine: a line-by-line port of the Python core (homepiggy/).
//
// Every function names the Python function it ports. Numerics follow the Python order of operations
// so that, fed the same random draws, both engines give the same paths (tests/test_web_parity.py).
// Methods are documented in docs/METHODS.md; nothing here adds a model.
//
// Scope of the port: step glidepaths (any table), baseline + historical-replay scenarios, the
// put-spread ladder without call overwrite (call_funding = 0, the product default), NL 2026 rules,
// box 3. Target-vol / lock-in glidepaths and the call overwrite stay Python-only.

export const SLEEVES = ["equity", "protected_equity", "short_bonds", "cash"];
export const LEVEL_VARS = ["short_rate", "term_spread", "mortgage_rate", "inflation"];
const IDX = { equity: 0, house: 1, short_rate: 2, term_spread: 3, mortgage_rate: 4, inflation: 5 };

export const DEFAULT_TABLE = [
  { from_years: 10, to_years: 7, weights: [0.80, 0.20, 0.00, 0.00] },
  { from_years: 7, to_years: 4, weights: [0.30, 0.60, 0.10, 0.00] },
  { from_years: 4, to_years: 2, weights: [0.00, 0.20, 0.30, 0.50] },
  { from_years: 2, to_years: 0, weights: [0.00, 0.00, 0.20, 0.80] },
];

export const PRODUCT = {          // sleeves.ProductParams defaults (design choices, not market data)
  equity_ter: 0.0020, protected_equity_ter: 0.0050, short_bonds_ter: 0.0015, cash_spread: 0.0010,
  bond_duration: 1.9, bond_tenor_years: 2.0, cash_tenor_years: 0.25,
  buffer_upper_strike: 1.00, buffer_lower_strike: 0.90, outcome_months: 12, n_tranches: 4, option_cost: 0.0020,
};
export const REBALANCE = { every_months: 3, drift_threshold_l1: 0.05, transaction_cost: 0.0005 };
export const BENCHMARK_SAVINGS_SPREAD = 0.0050;

export const DEFAULT_SAVER = {    // engine.SaverProfile + BuyerProfile defaults
  horizon_years: 10, initial_eur: 1000, monthly_contribution_eur: 550, gross_income_eur: 60000,
  real_wage_growth: 0.01, house_price_eur: 400000, student_debt_eur: 0, student_rate: 0.0233,
  student_repay_start_month: 48, student_term_months: 420, own_target_pct: 0.10, apply_income_test: false,
  age_at_start: 20, single: true, first_time_buyer: true, owner_occupier: true, fixed_period_years: 10,
  energy_label: "EFG", use_nhg: true, other_costs_eur: 0,
};

// ----------------------------------------------------------------------------- special functions
// Normal CDF via erf/erfc: Taylor series of erf for |x| < 2 (alternating terms, abs. error < 1e-15),
// Lentz continued fraction for erfc for |x| >= 2 (converges in < 60 terms there).
function erfSeries(x) {
  let sum = x, term = x;
  const x2 = x * x;
  for (let n = 1; n < 80; n++) {
    term *= -x2 / n;
    const add = term / (2 * n + 1);
    sum += add;
    if (Math.abs(add) < 1e-17 * Math.abs(sum)) break;
  }
  return (2 / Math.sqrt(Math.PI)) * sum;
}
function erfcCF(x) {              // x >= 2; erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + 1/2/(x + 1/(x + 3/2/(x + ...))))
  const tiny = 1e-300;
  let f = x, C = x, D = 0;
  for (let n = 1; n < 200; n++) {
    const a = n / 2;
    D = x + a * D; if (Math.abs(D) < tiny) D = tiny;
    C = x + a / C; if (Math.abs(C) < tiny) C = tiny;
    D = 1 / D;
    const delta = C * D;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-16) break;
  }
  return Math.exp(-x * x) / Math.sqrt(Math.PI) / f;
}
export function erfc(x) {
  if (x >= 2) return erfcCF(x);
  if (x <= -2) return 2 - erfcCF(-x);
  return 1 - erfSeries(x);
}
export function ndtr(a) {         // scipy.special.ndtr
  const x = a * Math.SQRT1_2;
  if (Math.abs(x) < Math.SQRT1_2) return 0.5 + 0.5 * erfSeries(x);
  const y = 0.5 * erfc(Math.abs(x));
  return x > 0 ? 1 - y : y;
}
function rintHalfEven(x) {        // numpy rint
  const r = Math.round(x);
  return (Math.abs(x % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r;
}
function round3(x) { return rintHalfEven(x * 1000) / 1000; }            // np.round(x, 3)
function searchLeft(arr, v) { let i = 0; while (i < arr.length && arr[i] < v) i++; return i; }
function searchRight(arr, v) { let i = 0; while (i < arr.length && arr[i] <= v) i++; return i; }

// ----------------------------------------------------------------------------- random numbers (browser only)
// sfc32 (Small Fast Counting PRNG, period ~2^128), 53-bit uniforms; polar-method normals.
// The parity test bypasses this and injects numpy's draws, so it does not affect model checks.
export function makeRng(seed, stream) {
  let a = (seed ^ 0x9e3779b9) >>> 0, b = (stream * 0x85ebca6b + 0x243f6a88) >>> 0, c = 0xb7e15162 ^ stream, d = 1;
  const next = () => {
    a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0;
    let t = (a + b) | 0; a = b ^ (b >>> 9); b = (c + (c << 3)) | 0; c = (c << 21) | (c >>> 11);
    d = (d + 1) | 0; t = (t + d) | 0; c = (c + t) | 0;
    return t >>> 0;
  };
  for (let i = 0; i < 20; i++) next();
  const uniform = () => (next() * 2097152 + (next() >>> 11)) / 9007199254740992;  // [0,1), 53 bits
  let spare = null;
  const normal = () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u, v, s;
    do { u = 2 * uniform() - 1; v = 2 * uniform() - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
    const m = Math.sqrt(-2 * Math.log(s) / s);
    spare = v * m; return u * m;
  };
  return { uniform, normal };
}
export function drawRandom(seed, nPaths, T) {     // engine._streams + simulate_market_paths draw shapes
  const gi = makeRng(seed, 1), gt = makeRng(seed, 2), gn = makeRng(seed, 3);
  const init = new Float64Array(nPaths), U = new Float64Array(nPaths * T), Z = new Float64Array(nPaths * T * 6);
  for (let i = 0; i < init.length; i++) init[i] = gi.uniform();
  for (let i = 0; i < U.length; i++) U[i] = gt.uniform();
  for (let i = 0; i < Z.length; i++) Z[i] = gn.normal();
  return { init, U, Z };
}

// ----------------------------------------------------------------------------- scenarios (scenarios.py)
export function historicalScenario(event, T, endMonthsBeforePurchase = 0) {
  const n = event.duration_months, end = T - endMonthsBeforePurchase, start = end - n;
  if (start < 0 || end > T) throw new Error(`Episode of ${n} months does not fit the horizon.`);
  const segments = [];
  let i = 0;
  while (i < n) { let j = i; while (j < n && event.regime_path[j] === event.regime_path[i]) j++;
    segments.push({ start: start + i, end: start + j, regime: event.regime_path[i] }); i = j; }
  return { name: event.name, segments, shocks: [{ month: start, duration_months: n, ...event.shocks }] };
}
function shockMatrix(shocks, T) {                  // market.shock_matrix -> (T, 6)
  const S = new Float64Array(T * 6);
  for (const ev of shocks || []) {
    const n = ev.duration_months;
    if (n < 1 || ev.month < 0 || ev.month + n > T) throw new Error("Shock outside the horizon.");
    for (let t = ev.month; t < ev.month + n; t++) {
      S[t * 6 + IDX.equity] += (ev.equity_log || 0) / n;
      S[t * 6 + IDX.house] += (ev.house_log || 0) / n;
      for (const v of LEVEL_VARS) S[t * 6 + IDX[v]] += (ev[v] || 0) / n;
    }
  }
  return S;
}

// ----------------------------------------------------------------------------- regimes + market
export function simulateRegimes(p, nP, T, initU, U, segments) {   // regimes.simulate_regimes
  const K = p.regimes.length, R = new Int8Array(nP * (T + 1));
  const cumInit = []; let acc = 0; for (const q of p.init_probs) { acc += q; cumInit.push(acc); }
  const cum = p.P.map(row => { let s = 0; const c = row.map(x => (s += x)); c[K - 1] = 1 + 1e-12; return c; });
  const force = new Int8Array(T).fill(-1);
  for (const sg of segments || []) {
    const k = p.regimes.indexOf(sg.regime);
    if (k < 0 || !(0 <= sg.start && sg.start < sg.end && sg.end <= T)) throw new Error("Bad scenario segment.");
    for (let t = sg.start; t < sg.end; t++) force[t] = k;
  }
  for (let i = 0; i < nP; i++) {
    R[i * (T + 1)] = Math.min(searchRight(cumInit, initU[i]), K - 1);
    for (let t = 0; t < T; t++) {
      let nxt;
      if (force[t] >= 0) nxt = force[t];
      else { const c = cum[R[i * (T + 1) + t]], u = U[i * T + t]; nxt = 0; while (!(u < c[nxt])) nxt++; }
      R[i * (T + 1) + t + 1] = nxt;
    }
  }
  return R;
}

export function simulateMarket(p, R, Z, nP, T, shocks) {          // market.simulate_market
  const T1 = T + 1, S = shockMatrix(shocks, T), sq12 = Math.sqrt(12);
  const eq = new Float64Array(nP * T), eqIndex = new Float64Array(nP * T1), houseIndex = new Float64Array(nP * T1);
  const lv = {}; for (const v of LEVEL_VARS) lv[v] = new Float64Array(nP * T1);
  const cpi = new Float64Array(nP * T1);
  const kappa = {}; for (const v of LEVEL_VARS) kappa[v] = 1 - Math.pow(0.5, 1 / p.levels[v].half_life_months);
  const muH = p.house.mu.map(x => x / 12), phi = p.house.ar1, z = new Float64Array(6);
  for (let i = 0; i < nP; i++) {
    const o = i * T1;
    for (const v of LEVEL_VARS) lv[v][o] = p.levels[v].initial;
    let gPrev = muH[R[o]], hLog = 0, eqCum = 0, cp = 1;
    eqIndex[o] = 1; houseIndex[o] = 1; cpi[o] = 1;
    for (let t = 0; t < T; t++) {
      const s = R[o + t + 1], L = p.chol[s], zo = (i * T + t) * 6;
      for (let a = 0; a < 6; a++) { let acc = 0; for (let b = 0; b <= a; b++) acc += L[a][b] * Z[zo + b]; z[a] = acc; }
      const x = p.equity.mu[s] / 12 + p.equity.sigma[s] / sq12 * z[IDX.equity] + S[t * 6 + IDX.equity];
      eq[i * T + t] = x;
      eqCum += x; eqIndex[o + t + 1] = Math.exp(eqCum);
      const g = muH[s] + phi * (gPrev - muH[R[o + t]]) + p.house.sigma[s] / sq12 * z[IDX.house];
      hLog = hLog + g + S[t * 6 + IDX.house]; houseIndex[o + t + 1] = Math.exp(hLog); gPrev = g;
      for (const v of LEVEL_VARS) {
        const pv = p.levels[v], prev = lv[v][o + t];
        lv[v][o + t + 1] = prev + kappa[v] * (pv.theta[s] - prev) + pv.sigma[s] / sq12 * z[IDX[v]] + S[t * 6 + IDX[v]];
      }
      cp *= 1 + lv.inflation[o + t + 1] / 12; cpi[o + t + 1] = cp;
    }
  }
  return { nP, T, regimes: R, equityLogret: eq, equityIndex: eqIndex, houseIndex, shortRate: lv.short_rate,
           termSpread: lv.term_spread, mortgageRate: lv.mortgage_rate, inflation: lv.inflation, cpi };
}

// ----------------------------------------------------------------------------- sleeves (sleeves.py, options.py)
function impliedVol(atm, m, skew, minVol) { const safe = m > 0 ? m : 1; return Math.max(minVol, atm + skew * (-Math.log(safe))); }
export function bsPut(S, K, tau, rSimple, sigma) {                  // options.bs_price(kind="put")
  const intrinsic = Math.max(K - S, 0);
  if (!(tau > 1e-12 && K > 0)) return intrinsic;
  const r = Math.log1p(Math.max(rSimple, -0.99)), v = Math.max(sigma, 1e-8);
  const sqt = v * Math.sqrt(tau), d1 = (Math.log(S / K) + (r + 0.5 * v * v) * tau) / sqt, d2 = d1 - sqt;
  return K * Math.exp(-r * tau) * ndtr(-d2) - S * ndtr(-d1);
}

function sleeveReturns(m, prod) {                  // equity_returns, cash_returns, bond_returns
  const { nP, T } = m, T1 = T + 1;
  const eqR = new Float64Array(nP * T), cashR = new Float64Array(nP * T), bondR = new Float64Array(nP * T);
  const w = (1 - prod.cash_tenor_years) / (prod.bond_tenor_years - prod.cash_tenor_years);
  for (let i = 0; i < nP; i++) for (let t = 0; t < T; t++) {
    const k = i * T + t, o = i * T1 + t;
    eqR[k] = Math.expm1(m.equityLogret[k]) - prod.equity_ter / 12;
    cashR[k] = (m.shortRate[o] - prod.cash_spread) / 12;
    const r0 = m.shortRate[o], y0 = r0 + m.termSpread[o], y1n = m.shortRate[o + 1] + m.termSpread[o + 1];
    const y1 = r0 + (y0 - r0) * w, slope = (y0 - y1) / (prod.bond_tenor_years - 1);
    bondR[k] = (y0 + prod.bond_duration * slope) / 12 + (-prod.bond_duration * (y1n - y0)) - prod.short_bonds_ter / 12;
  }
  return { eqR, cashR, bondR };
}

class ProtectedEquitySleeve {                      // sleeves.ProtectedEquitySleeve (call_funding = 0)
  constructor(m, p, prod) {
    this.m = m; this.p = p; this.prod = prod; this.L = prod.n_tranches; this.t = 0;
    const nP = m.nP, L = this.L;
    this.n = new Float64Array(nP * L); this.K1 = new Float64Array(nP * L); this.K2 = new Float64Array(nP * L);
    const step = prod.outcome_months / L;
    this.expiry = Array.from({ length: L }, (_, j) => (j + 1) * step);
    this.u = new Float64Array(nP * L); this.uT = -1;
    for (let j = 0; j < L; j++) this._strike(j, 0, this.expiry[j]);
  }
  _atm(i, t) { return this.p.implied_vol.vrp_ratio * this.p.equity.sigma[this.m.regimes[i * (this.m.T + 1) + t]]; }
  _unit(i, j, t) {
    const o = i * (this.m.T + 1) + t, S = this.m.equityIndex[o], r = this.m.shortRate[o];
    const tau = Math.max(this.expiry[j] - t, 0) / 12, iv = this.p.implied_vol, atm = this._atm(i, t);
    const K1 = this.K1[i * this.L + j], K2 = this.K2[i * this.L + j];
    const p1 = bsPut(S, K1, tau, r, impliedVol(atm, K1 / S, iv.skew, iv.min_vol));
    const p2 = bsPut(S, K2, tau, r, impliedVol(atm, K2 / S, iv.skew, iv.min_vol));
    return S + p1 - p2;
  }
  _units(t) {
    if (this.uT === t) return this.u;
    for (let i = 0; i < this.m.nP; i++) for (let j = 0; j < this.L; j++) this.u[i * this.L + j] = this._unit(i, j, t);
    this.uT = t; return this.u;
  }
  _strike(j, t, tenor) {
    const prod = this.prod, iv = this.p.implied_vol, tau = tenor / 12, k1 = prod.buffer_upper_strike, k2 = prod.buffer_lower_strike;
    const cost = new Float64Array(this.m.nP);
    for (let i = 0; i < this.m.nP; i++) {
      const o = i * (this.m.T + 1) + t, S = this.m.equityIndex[o], r = this.m.shortRate[o], atm = this._atm(i, t);
      const p1 = bsPut(S, k1 * S, tau, r, impliedVol(atm, k1, iv.skew, iv.min_vol));
      const p2 = bsPut(S, k2 * S, tau, r, impliedVol(atm, k2, iv.skew, iv.min_vol));
      this.K1[i * this.L + j] = k1 * S; this.K2[i * this.L + j] = k2 * S;
      cost[i] = S + (p1 - p2) - 0 + prod.option_cost * S;
    }
    this.expiry[j] = t + tenor; this.uT = -1;
    return cost;
  }
  advance(t, out) {
    if (t !== this.t + 1) throw new Error("advance one month at a time");
    this.t = t; const nP = this.m.nP, L = this.L, f = 1 - this.prod.protected_equity_ter / 12;
    for (let k = 0; k < this.n.length; k++) this.n[k] *= f;
    for (let j = 0; j < L; j++) {
      if (this.expiry[j] !== t) continue;
      const V = new Float64Array(nP);
      for (let i = 0; i < nP; i++) V[i] = this.n[i * L + j] * this._unit(i, j, t);
      if (t < this.m.T) {
        const cost = this._strike(j, t, this.prod.outcome_months);
        for (let i = 0; i < nP; i++) this.n[i * L + j] = V[i] / cost[i];
      } else {
        for (let i = 0; i < nP; i++) {
          this.K1[i * L + j] = 0; this.K2[i * L + j] = 0;
          this.n[i * L + j] = V[i] / this.m.equityIndex[i * (this.m.T + 1) + t];
        }
        this.uT = -1;
      }
    }
    const u = this._units(t);
    for (let i = 0; i < nP; i++) { let s = 0; for (let j = 0; j < L; j++) s += this.n[i * L + j] * u[i * L + j]; out[i] = s; }
  }
  setValue(i, target) {                            // per path (Python vectorises the same arithmetic)
    target = Math.max(target, 0); const L = this.L, u = this._units(this.t);
    let cur = 0; for (let j = 0; j < L; j++) cur += this.n[i * L + j] * u[i * L + j];
    const has = cur > 1e-12, scale = has ? target / cur : 0;
    for (let j = 0; j < L; j++) this.n[i * L + j] *= scale;
    if (!has && target > 0) for (let j = 0; j < L; j++) this.n[i * L + j] = (target / L) / u[i * L + j];
  }
}

// ----------------------------------------------------------------------------- glidepath (glidepath.py, step kind)
export function staticWeights(table, T) {
  const segs = table.map(r => ({ from: r.from_years * 12, to: r.to_years * 12, w: r.weights.slice() }))
    .sort((a, b) => b.from - a.from);
  if (Math.abs(segs[segs.length - 1].to) > 1e-9) throw new Error("Glidepath table must end at 0 years.");
  for (const s of segs) { const sum = s.w.reduce((a, b) => a + b, 0);
    if (Math.abs(sum - 1) > 1e-9 || s.w.some(x => x < -1e-12)) throw new Error("Weights must be >= 0 and sum to 1."); }
  for (let k = 1; k < segs.length; k++) if (Math.abs(segs[k - 1].to - segs[k].from) > 1e-9) throw new Error("Glidepath gap/overlap.");
  if (segs[0].from + 1e-9 < T) throw new Error("Glidepath table starts after the horizon.");
  const W = [];
  for (let t = 0; t <= T; t++) {
    const x = T - t; let w = segs[0].w;
    for (const s of segs) if ((s.to < x && x <= s.from) || (x <= 0 && s.to === 0)) { w = s.w; break; }
    const sum = w.reduce((a, b) => a + b, 0); W.push(w.map(v => Math.max(v, 0) / sum));
  }
  return W;
}

// ----------------------------------------------------------------------------- box 3 (tax_nl.py)
function box3Tax(bank, other, b3) {
  const G = bank + other, R = bank * b3.f_bank + other * b3.f_other;
  const pct = G > 0 ? R / G : 0, B = Math.max(G - b3.allowance, 0);
  const total = b3.tax_rate * Math.max(pct * B, 0), share = G > 0 ? (bank + other) / G : 0;
  return total * share;
}

// ----------------------------------------------------------------------------- portfolio (portfolio.py)
function runPortfolio(rets, pe, W, nP, T, initial, contrib, b3, reb, bankMask, keepPath) {
  const K = 4, PE = 1, T1 = T + 1;
  let cur = new Float64Array(nP * K);
  const value = keepPath ? new Float64Array(nP * T1) : null, finalV = new Float64Array(nP);
  const peVal = new Float64Array(nP), w0 = W[0];
  for (let i = 0; i < nP; i++) {
    for (let k = 0; k < K; k++) cur[i * K + k] = initial * w0[k];
    if (pe) pe.setValue(i, cur[i * K + PE]);
    if (value) value[i * T1] = initial;
  }
  const names = ["eqR", null, "bondR", "cashR"];
  for (let t = 1; t <= T; t++) {
    if (pe) pe.advance(t, peVal);
    const wt = W[t];
    for (let i = 0; i < nP; i++) {
      const o = i * K;
      for (let k = 0; k < K; k++) { if (k === PE) continue; cur[o + k] *= 1 + rets[names[k]][i * T + t - 1]; }
      if (pe) cur[o + PE] = peVal[i];
      let total = cur[o] + cur[o + 1] + cur[o + 2] + cur[o + 3];
      if (b3.enabled && t % 12 === 0) {
        let bank = 0, other = 0;
        for (let k = 0; k < K; k++) { if (bankMask[k]) bank += cur[o + k]; else other += cur[o + k]; }
        const due = Math.min(box3Tax(bank, other, b3), total), frac = total > 0 ? due / total : 0;
        for (let k = 0; k < K; k++) cur[o + k] *= (1 - frac);
        total = cur[o] + cur[o + 1] + cur[o + 2] + cur[o + 3];
      }
      const c = contrib[i * T1 + t];
      for (let k = 0; k < K; k++) cur[o + k] += c * wt[k];
      total = cur[o] + cur[o + 1] + cur[o + 2] + cur[o + 3];
      let drift = 0;
      for (let k = 0; k < K; k++) { const wc = total > 0 ? cur[o + k] / total : wt[k]; drift += Math.abs(wc - wt[k]); }
      if (t % reb.every_months === 0 || drift > reb.drift_threshold_l1) {
        let traded = 0; const nw = [0, 0, 0, 0];
        for (let k = 0; k < K; k++) { nw[k] = total * wt[k]; traded += Math.abs(nw[k] - cur[o + k]); }
        const cost = reb.transaction_cost * traded, f = total > 0 ? 1 - cost / total : 0;
        for (let k = 0; k < K; k++) cur[o + k] = nw[k] * f;
      }
      if (pe) pe.setValue(i, cur[o + PE]);
      const v = cur[o] + cur[o + 1] + cur[o + 2] + cur[o + 3];
      if (value) value[i * T1 + t] = v;
      if (t === T) finalV[i] = v;
    }
  }
  return { value, finalV };
}

// ----------------------------------------------------------------------------- NL affordability (affordability_nl.py)
function annuityFactor(i, n) { return Math.abs(i) < 1e-10 ? n : (1 - Math.pow(1 + i, -n)) / i; }
function annuityPayment(principal, rate, n) { return principal / annuityFactor(rate / 12, n); }
function studentInstalments(s, T) {               // engine._student_instalment
  let B = s.student_debt_eur, pay = null; const out = new Float64Array(T + 1);
  for (let t = 0; t <= T; t++) {
    if (t > 0) { B *= 1 + s.student_rate / 12;
      if (t > s.student_repay_start_month) { if (pay === null) pay = annuityPayment(B, s.student_rate, s.student_term_months); B = Math.max(B - pay, 0); } }
    const repaid = Math.max(t - s.student_repay_start_month, 0), remaining = Math.max(s.student_term_months - repaid, 1);
    out[t] = B > 0 ? annuityPayment(B, s.student_rate, remaining) : 0;
  }
  return out;
}
export function affordability(price, income, m, instal, age, cpiF, houseF, s, pol) {
  const tr = pol.test_rate;
  const test = s.fixed_period_years >= tr.short_fixed_period_threshold_years ? m : Math.max(m, tr.short_fixed_period_minimum_rate);
  const inc26 = income / cpiF, rPct = round3(test * 100);
  const col = searchLeft(pol.flp_rate_upper_pct, rPct);
  const row = Math.min(Math.max(searchRight(pol.flp_income_rows, inc26) - 1, 0), pol.flp_income_rows.length - 1);
  const flp = pol.flp_pct[row][col];
  const gross = pol.grossup_factors[searchLeft(pol.grossup_upper_pct, rPct)];
  let budget = income * flp / 12 - instal * gross; budget = Math.max(budget, 0);
  let loan = budget * annuityFactor(test / 12, pol.annuity_term_months);
  const sp = pol.single_person_extra_loan;
  if (s.single) loan = loan + (inc26 > sp.min_toetsinkomen_eur ? sp.amount_eur * cpiF : 0);
  loan = loan + pol.energy_label_extra_loan_eur[s.energy_label] * cpiF;
  const ltvLoan = pol.max_ltv * price;
  const maxLoan = s.apply_income_test ? Math.min(loan, ltvLoan) : ltvLoan;   // income test optional (engine.SaverProfile)
  const se = pol.transfer_tax.starters_exemption;
  const exempt = se.min_age <= age && age < se.max_age_exclusive && s.first_time_buyer && s.owner_occupier
    && price <= se.house_value_limit_eur * houseF;
  const rate = s.owner_occupier ? pol.transfer_tax.owner_occupier_rate : pol.transfer_tax.non_owner_occupier_rate;
  const tt = exempt ? 0 : rate * price;
  const nhg = s.use_nhg && maxLoan <= pol.nhg.cost_limit_eur * houseF ? pol.nhg.one_off_premium_rate * maxLoan : 0;
  const other = s.other_costs_eur * cpiF;
  const gap = Math.max(price + tt + nhg + other - maxLoan, 0);
  return { test, flp, incomeLoan: loan, maxLoan, tt, nhg, other, gap };
}

// ----------------------------------------------------------------------------- full run (engine.run)
/**
 * opts: { nPaths, seed, table, scenario, draws, keepPaths, months: [t...] }
 *   draws: {init, U, Z} to replay external random numbers (parity tests); otherwise drawn from seed.
 *   keepPaths: also return per-month pot, target, legal minimum and bank pot (needed for fan charts).
 * Returns final-month arrays V (pot), G (own target), D (legal minimum), B (bank savings pot).
 */
export function run(params, policy, saverIn, opts) {
  const r = runMany(params, policy, saverIn, { ...opts, tables: [opts.table || DEFAULT_TABLE] });
  const f = r.strategies[0];
  return { ...r, V: f.V, W: f.W, Vpath: f.Vpath };
}

/** Same market paths (common random numbers) and the same targets for several glidepath tables. */
export function runMany(params, policy, saverIn, opts) {
  const s = { ...DEFAULT_SAVER, ...saverIn }, prod = PRODUCT, T = s.horizon_years * 12, T1 = T + 1;
  const nP = opts.nPaths, sc = opts.scenario || { segments: [], shocks: [] };
  const draws = opts.draws || drawRandom(opts.seed ?? 20261110, nP, T);
  const R = simulateRegimes(params, nP, T, draws.init, draws.U, sc.segments);
  const m = simulateMarket(params, R, draws.Z, nP, T, sc.shocks);

  const income = new Float64Array(nP * T1), contrib = new Float64Array(nP * T1);
  for (let i = 0; i < nP; i++) for (let t = 0; t <= T; t++) {
    const idx = m.cpi[i * T1 + t] * Math.pow(1 + s.real_wage_growth, t / 12);
    income[i * T1 + t] = s.gross_income_eur * idx;
    contrib[i * T1 + t] = t === 0 ? 0 : s.monthly_contribution_eur * idx;
  }
  const instal = studentInstalments(s, T);
  // measure 2: own funds needed WITH the income test (same paths), at purchase only
  const Dinc = new Float64Array(nP), Ginc = new Float64Array(nP);
  for (let i = 0; i < nP; i++) {
    const o = i * T1 + T, price = s.house_price_eur * m.houseIndex[o];
    const a = affordability(price, income[o], m.mortgageRate[o], instal[T], s.age_at_start + T / 12, m.cpi[o], m.houseIndex[o],
                            { ...s, apply_income_test: true }, policy);
    Dinc[i] = a.gap; Ginc[i] = Math.max(a.gap, s.own_target_pct * price + (a.tt + a.nhg + a.other));
  }
  const months = opts.keepPaths ? [...Array(T1).keys()] : [T];
  const D = {}, G = {};
  for (const t of months) {
    const d = new Float64Array(nP), g = new Float64Array(nP);
    for (let i = 0; i < nP; i++) {
      const o = i * T1 + t, price = s.house_price_eur * m.houseIndex[o];
      const a = affordability(price, income[o], m.mortgageRate[o], instal[t], s.age_at_start + t / 12, m.cpi[o], m.houseIndex[o], s, policy);
      d[i] = a.gap; g[i] = Math.max(a.gap, s.own_target_pct * price + (a.tt + a.nhg + a.other));
    }
    D[t] = d; G[t] = g;
  }

  const b3 = { enabled: true, tax_rate: policy.box3.tax_rate, allowance: policy.box3.allowance_single_eur,
               f_bank: policy.box3.deemed_return.bank_deposits, f_other: policy.box3.deemed_return.other_assets };
  const rets = sleeveReturns(m, prod);
  const strategies = opts.tables.map(table => {
    const W = staticWeights(table, T);
    const pe = W.some(w => w[1] > 0) ? new ProtectedEquitySleeve(m, params, prod) : null;
    const port = runPortfolio(rets, pe, W, nP, T, s.initial_eur, contrib, b3, REBALANCE, [false, false, false, false], opts.keepPaths);
    return { W, V: port.finalV, Vpath: port.value, irr: irrAnnual(contrib, s.initial_eur, port.finalV, nP, T) };
  });

  const bankR = { eqR: new Float64Array(nP * T), bondR: new Float64Array(nP * T), cashR: new Float64Array(nP * T) };
  for (let i = 0; i < nP; i++) for (let t = 0; t < T; t++) bankR.cashR[i * T + t] = (m.shortRate[i * T1 + t] - BENCHMARK_SAVINGS_SPREAD) / 12;
  const cashW = Array.from({ length: T1 }, () => [0, 0, 0, 1]);
  const bench = runPortfolio(bankR, null, cashW, nP, T, s.initial_eur, contrib, b3, { ...REBALANCE, transaction_cost: 0 },
                             [false, false, false, true], opts.keepPaths);

  const housePa = new Float64Array(nP);
  for (let i = 0; i < nP; i++) housePa[i] = Math.pow(m.houseIndex[i * T1 + T], 12 / T) - 1;
  const out = { T, nPaths: nP, strategies, B: bench.finalV, G: G[T], D: D[T], Dinc, Ginc, housePa,
                bankIrr: irrAnnual(contrib, s.initial_eur, bench.finalV, nP, T), market: m, saver: s, instal };
  if (opts.keepPaths) { out.Bpath = bench.value; out.Gpath = G; out.Dpath = D; out.income = income; }
  return out;
}

// ----------------------------------------------------------------------------- statistics (metrics.py)
/** metrics.irr_monthly by bisection on [-0.2, 0.2] per month (80 steps), annualised; NaN without a sign change. */
export function irrAnnual(contrib, initial, finalV, nP, T) {
  const T1 = T + 1, out = new Float64Array(nP), f = new Float64Array(T1);
  const npv = (i) => { let s = 0; for (let t = 0; t <= T; t++) s += f[t] / Math.pow(1 + i, t); return s; };
  for (let p = 0; p < nP; p++) {
    for (let t = 0; t <= T; t++) f[t] = -(t === 0 ? initial : contrib[p * T1 + t]);
    f[T] += finalV[p];
    let a = -0.2, b = 0.2, fa = npv(a);
    const fb = npv(b), ok = Math.sign(fa) !== Math.sign(fb);
    for (let k = 0; k < 80; k++) {
      const mid = 0.5 * (a + b), fm = npv(mid);
      if (Math.sign(fm) === Math.sign(fa)) { a = mid; fa = fm; } else { b = mid; }
    }
    out[p] = ok ? Math.pow(1 + 0.5 * (a + b), 12) - 1 : NaN;
  }
  return out;
}
/** Lower-tail expected shortfall: mean of the worst ceil(alpha * n) values (glidepath_opt.cvar_low). */
export function cvarLow(x, alpha = 0.10) {
  const a = Float64Array.from(x).sort(), k = Math.max(1, Math.ceil(alpha * a.length));
  let s = 0; for (let i = 0; i < k; i++) s += a[i]; return s / k;
}
export function quantile(arr, q) {                 // numpy linear (type 7)
  const a = Float64Array.from(arr).sort(), h = (a.length - 1) * q, lo = Math.floor(h);
  return lo + 1 < a.length ? a[lo] + (h - lo) * (a[lo + 1] - a[lo]) : a[lo];
}
export function prob(mask) { let s = 0; for (const x of mask) s += x ? 1 : 0; return s / mask.length; }
export function wilson(k, n, z = 1.959963984540054) {       // metrics.prob (Wilson 1927)
  if (n === 0) return [NaN, NaN];
  const p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}
