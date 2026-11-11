# Methods

This document declares every statistical and financial method used by the
`homepiggy` Python core, with its equation, its assumptions, and why it was
chosen. Where the implementation deviates from the design agreed in
conversation, the deviation and its reason are listed in
[section 9](#9-deviations-from-the-agreed-design).

**Numbers only.** Every computation is deterministic code fed by sourced
data files. No language model and no discretionary judgement enters any
calculation. Parameters are either estimated from cited data (the calibration
step, step 2) or are explicit user inputs (saver profile, product design).

Units: rates and returns are decimals (0.03 = 3%). "Annual" parameters are
converted to monthly in exactly one place (`market.py`): mean / 12,
volatility / sqrt(12).

---

## 1. Macro regimes: Markov chain (`regimes.py`)

* **State.** s_t in {Expansion, Late-cycle, Recession, Recovery}, the four
  regimes of the agentic-architecture repo (`MACRO_REGIMES`).
* **Dynamics.** A first-order, time-homogeneous Markov chain with a monthly
  transition matrix P, where P[i, j] = Pr(s_{t+1} = j | s_t = i). Rows are
  validated: square, finite, non-negative, summing to 1. Validation fails
  closed.
* **Sampling.** Inverse CDF on each row (exact for a categorical
  distribution), driven by externally supplied uniforms.
* **Diagnostics.**
  * The stationary distribution pi solves pi P = pi, sum pi = 1, as a linear
    system; a reducible chain is rejected.
  * Expected regime duration is 1 / (1 - P_ii), since sojourn times are
    geometric.
* **Narrative forcing.** A scenario may fix the regime over chosen month
  ranges. The same uniforms are consumed in every month, so paths under
  different scenarios are identical before the first forced month (common
  random numbers).
* **Estimation (step 2, done).** P is estimated by maximum likelihood
  (row-normalised transition counts) from historical regime labels, under two
  labelling methods kept side by side: the repo's threshold classifier applied
  to euro-area data, and a 4-state Gaussian hidden Markov model as a
  cross-check. See [CALIBRATION.md](CALIBRATION.md).

**Why a Markov chain:** it is the simplest model that produces regime
persistence and random timing, both essential for a 10-year horizon. Each
entry can be estimated directly from transition counts.

## 2. Market state variables (`market.py`)

The innovation vector z_t ~ N(0, R[s_t]) is built as z = L e, with
e ~ iid N(0, I) and L the Cholesky factor of R[s_t]. Each R is first repaired
to the nearest positive-definite correlation matrix by eigenvalue clipping,
then rescaled to a unit diagonal. This is the repo's `enforce_psd` repair.

| Variable | Equation (monthly) | Notes |
|---|---|---|
| Equity log return | x_t = mu_e[s]/12 + sigma_e[s]/sqrt(12) z_eq | global equity, EUR, total return |
| House log growth | g_t = mu_h[s]/12 + phi (g_{t-1} - mu_h[s_{t-1}]/12) + sigma_h[s]/sqrt(12) z_h | AR(1) momentum, see section 9 |
| Short rate, term spread (2y - short), mortgage rate, inflation | y_t = y_{t-1} + kappa (theta[s] - y_{t-1}) + sigma[s]/sqrt(12) z_y | regime-switching discretised Ornstein-Uhlenbeck (Vasicek) |

* kappa = 1 - 0.5^(1/half_life_months): a deviation from theta halves in
  `half_life_months`. This is tested exactly.
* **Mortgage rates move with the regime:** each regime has its own long-run
  mortgage level theta_m[s], and mortgage innovations are correlated with the
  short rate, the term spread and inflation.
* Rates are not floored, because euro rates were negative from 2014 to 2022.
  The affordability rules handle low and negative test rates correctly
  (annuity limit, lowest table bracket).
* CPI_t = CPI_{t-1} (1 + pi_t / 12).
* **Shocks (narrative scenarios)** are deterministic and spread evenly over
  their duration:
  * equity and house shocks add to the log *level*, so a -20% shock is
    exactly -20%;
  * rate and inflation shocks add to the level and then decay through the
    AR(1).

**Why Gaussian within a regime:** the mixture across regimes already gives
fat tails and volatility clustering (Hamilton 1989; Ang & Bekaert 2002).
Tail events beyond that are added explicitly through narrative shocks rather
than through a distributional assumption.

## 3. Sleeves (`sleeves.py`, `options.py`)

### 3.1 Equity
R_t = exp(x_t) - 1 - TER/12.

### 3.2 Cash
R_t = (r_{t-1} - spread)/12. The rate is set at the start of the month.

### 3.3 Protected equity: put-spread buffer ladder
* **Structure per unit:** 1 index unit + long put (K1 = 100% of S at reset)
  − short put (K2 = 85–90%) − optional short call Kc.
  * At expiry the index loss between 0% and (1 − K2) is absorbed; losses
    beyond K2 pass through.
  * Tested exactly: an index loss of 20% with K2 = 90% gives a 10% loss on
    the notional.
* **Call overwrite (optional):** `call_funding` f in [0, 1]. Kc is set so that
  C(Kc) = f × (P(K1) − P(K2)), solved by bisection. The call price is strictly
  decreasing in K, so the root is unique. f = 0 is the pure put spread you
  specified and is the default.
* **Pricing:** Black-Scholes on the total-return index (q = 0), with
  r = ln(1 + short rate).
  * Implied volatility: sigma(K) = max(min_vol, vrp × sigma_e[s] + skew × (−ln K/S)).
  * Protection is dearer in high-volatility regimes, through sigma_e[s].
  * The volatility risk premium `vrp` and the `skew` are calibration
    parameters, estimated from VSTOXX/VIX and option-surface data in step 2.
* **Ladder:** L tranches (default 4), 12-month outcome periods.
  * First expiries are at 3, 6, 9 and 12 months, so the first periods are
    shorter. After that each tranche rolls every 12 months, struck at the
    then-current index level.
  * Option trading cost: `option_cost` × notional at each reset.
* **Mark to market and flows:** every tranche is revalued monthly with
  Black-Scholes. Contributions and rebalancing buy or sell pro rata across
  tranches at mark-to-market value, so a new investor is never stuck with a
  single reset date. Value per unit is S + P1 − P2 − C > 0, so the sleeve
  cannot go negative.

### 3.4 Short bonds (1–3y government)
Monthly return = carry + roll-down − duration × yield change − TER/12:

R_t = (y2_{t-1} + D × slope_{t-1})/12 − D (y2_t − y2_{t-1}) − TER/12

* This is the repo's fixed-income building block (yield + roll-down;
  `run_fi_cma_builder`), applied month by month.
* slope = (y2 − y1) per year, with y1 linearly interpolated between the 3m
  rate and the 2y yield (the repo's linear interpolation).
* Roll-down for a constant-maturity fund is D × slope (section 9).
* Credit spread and default loss are zero because the sleeve holds
  government bonds.
* Convexity is omitted, as in the repo; the error is below 0.25 bp a month
  for D ≈ 1.9.

## 4. Glidepaths and portfolio (`glidepath.py`, `portfolio.py`)

### 4.1 Strategies
Weights over (equity, protected equity, short bonds, cash) as a function of
months-to-purchase tau.

* **step:** piecewise-constant table. Validation fails closed on gaps,
  overlaps, negative weights or rows not summing to 1. The default is the
  agreed path:

  | Years to purchase | Equity | Protected equity | Short bonds | Cash |
  |---|---|---|---|---|
  | 10–7 | 80% | 20% | – | – |
  | 7–4 | 30% | 60% | 10% | – |
  | 4–2 | – | 20% | 30% | **50%** |
  | 2–0 | – | – | 20% | **80%** |

* **linear:** interpolation between segment midpoints.
* **time_warp:** the table evaluated at tau' = H (tau/H)^gamma. gamma < 1
  de-risks later; gamma > 1 de-risks earlier.
* **target_vol:** w = lambda × w_base + (1 − lambda) × cash, with lambda in
  [0, 1].
  * lambda solves vol(w) = target(t) by bisection, where target(t) declines
    linearly.
  * vol uses the regime-conditional sleeve covariance, estimated by Monte
    Carlo moment estimation: 2,000 paths × 36 months per forced regime, the
    first 12 months discarded so the ladder is in steady state.
  * It never adds risk relative to the base table.
* **lock_in:** follow the table until pot ≥ multiple × required deposit
  (evaluated at month t with that month's price, mortgage rate and income),
  then hold `lock_weights` permanently.

### 4.2 Monthly accounting order
1. Apply sleeve returns; protected-equity tranches roll at their reset dates.
2. Box 3 tax at each 12-month anniversary.
3. Invest the contribution at target weights.
4. Rebalance to target if t is a multiple of 3 (quarterly) or the L1 drift
   Σ|w − w*| exceeds 5%. This is the repo's `REBALANCING_POLICY`.
   Transaction cost is tc × traded value.

The time-weighted NAV uses the pre-flow return, so contributions cannot mask
losses.

### 4.3 Saver cash flows
* Income_t = income_0 × CPI_t × (1 + real wage growth)^(t/12).
  Contributions scale with the same index (constant savings rate).
* House price_t = price_0 × house index_t.
* Benchmark: identical contributions into a bank savings account earning the
  short rate minus a spread. It runs on the *same paths*, so "outpace cash"
  is a paired comparison.

## 5. NL affordability gap (`affordability_nl.py`)

Required deposit D = max(price + costs − L, 0), where L is the maximum
mortgage:

1. **Test rate (toetsrente):** the mortgage rate for a fixed period of 10
   years or more; otherwise max(rate, 5%), the AFM toetsrente.
2. **Financing-cost % (FLP):** 2026 table cell (toetsinkomen row, toetsrente
   column). The test rate is rounded to 3 decimals in %, per the regulation's
   explanatory note. The 97 × 12 table is machine-extracted from the
   Staatscourant (`tools/extract_flp_table.py`), and its hash is pinned in
   the tests.
3. **Monthly budget:** income × FLP / 12 − DUO instalment × gross-up factor
   (Art. 3a). If DUO repayment hasn't started, the instalment is the annuity
   on the actual debt, rate and remaining term (Art. 3a lid 3).
4. **Income loan:** budget × annuity factor(test rate / 12, 360), plus
   €17,000 for singles with toetsinkomen > €30,000, plus the energy-label
   amount.
5. **LTV cap:** L = min(income loan, 100% × price).
6. **Costs:**
   * Transfer tax: 0 if the starter exemption applies (18 ≤ age < 35, price
     ≤ limit, first use, owner-occupier); otherwise 2% of the *whole* price.
   * NHG premium: 0.4% of the loan if the loan ≤ NHG limit (optional).
   * Other buyer costs: user input.

**Optional own savings target (user setting, default 10%).** Besides the legal
minimum D, the saver can set an own goal: G = max(D, X% × price at purchase +
buying costs). It is reported as a separate outcome, P(pot ≥ G), and the
lock-in glidepath locks when the pot reaches G. It matters most for high
earners, where the legal minimum shrinks to the buying costs. X is the saver's
choice, not a regulation.

**Projection to the purchase date (assumption):** the 2026 rules are held
constant in real terms.
* Euro thresholds (income brackets, singles amount and threshold, energy
  amounts) are scaled by simulated CPI.
* The starter and NHG price limits are scaled by the simulated house index.

## 6. Tax (`tax_nl.py`)

### 6.1 Scope
Only Dutch box 3 applies to the fund. The cash benchmark is taxed under the
same rules (bank-deposit category).

### 6.2 Box 3 (2026 deemed-return system)
1. Base G = bank + other.
2. Deemed return R = 1.28% × bank + 6.00% × other.
3. Effective rate p = R / G.
4. Taxable base B = max(G − €59,357, 0).
5. Tax = 36% × p × B.

* The fund's money-market sleeve counts as "other assets" by default. This
  is configurable, because a fund holding is not a bank deposit.
* Model simplifications: anniversaries stand in for 1 January; tax is paid
  immediately; the allowance is held at its nominal 2026 value. The
  tegenbewijsregeling is not modelled, so the tax is an upper bound.

## 7. Outcome statistics and uncertainty (`metrics.py`)

Every reported number carries its Monte Carlo uncertainty.

| Quantity | Estimator | Interval |
|---|---|---|
| P(reach deposit), P(outpace cash), P(keep up with housing), P(all three) | sample proportion | Wilson score 95% (Wilson 1927) |
| Pot, deposit, surplus and drawdown percentiles | sample quantile | distribution-free order-statistic 95% (binomial) |
| Expected shortfall E[max(D − V, 0)] | sample mean | CLT, s/sqrt(n) |
| Strategy A vs B | paired per-path differences on common random numbers | CLT on the differences |

* **Money-weighted return:** IRR on (−initial, −contributions, +final value).
  These flows change sign once, so the root is unique (Descartes' rule of
  signs); found by bisection.
* **"Keeps up with housing":** IRR ≥ annualised house-price growth over the
  same horizon.
* **Drawdown:** maximum drawdown of the time-weighted NAV over the final 24
  months, which measures sequence risk.

## 8. Randomness and reproducibility (`engine.py`)

* One integer seed feeds `numpy.random.SeedSequence`, which spawns
  independent PCG64 streams for the initial regime, the regime transitions
  and the market innovations.
* The same seed gives bit-identical results (tested).
* Every glidepath in `compare()` sees the same market paths (common random
  numbers), so strategy differences are not polluted by sampling noise.

## 9. Deviations from the agreed design

| Agreed | Implemented | Why |
|---|---|---|
| Each regime sets mean / vol / correlation of every variable | Same, but *levels* (rates, inflation) mean-revert to a regime-specific level instead of being drawn iid | Rates are levels, not returns. iid draws would make them jump between regime means every month, which is unrealistic and would overstate bond losses. |
| House-price growth from the regime mean / vol | Adds AR(1) persistence (phi) in *latent* growth, estimated in step 2 by Kalman-filter maximum likelihood with index measurement noise; phi = 0 recovers the iid case | Housing returns are strongly autocorrelated (Case & Shiller 1989). The CBS monthly index contains sampling noise that hides this (lag-1 autocorrelation 0.36 < lag-2 0.50), and a plain AR(1) covered only 13% of historical 10-year windows. The state-space model (with the mean constraint below) raises that to 100% under both regime methods (30 overlapping windows). |
| House-price long-run mean by likelihood | Frequency-weighted regime mean fixed to the sample mean (method of moments); regime differences, persistence and noise by Kalman maximum likelihood | With phi near 1 the likelihood estimate of the level was biased (+2.3 pp p.a. vs the 3.7% sample mean). The sample mean is the robust estimator of a stationary process's mean. |
| Calibration sample "from 1999" | Regime-conditional estimates use 2002-02 to 2026-09 | The repo classifier needs 24 months of indicator history, and the euro-area unemployment series starts 2000-03. |
| Rate regime levels from OLS | Constrained least squares: each regime's long-run level must lie inside the observed range | Over 2002–2026 rates behave almost like a random walk, so unconstrained regime levels came out at 35% / −12%. Binding constraints are reported. |
| One regime classifier | Two (repo classifier + HMM), both calibrated and bootstrapped | You asked for both. They agree on only 38.5% of months (κ = 0.21), so results are shown under each. |
| Bond roll-down "by rolling 1 year down the curve" (repo) | Roll-down = D × slope for a constant-maturity fund | A 1–3y index fund keeps its maturity roughly constant. D × slope is the continuous-time roll for such a fund; the repo's formula applies to a single bond held for one year. |
| Options on the equity index | Options on the **total-return** index (q = 0) | Keeps dividends inside the protected notional and avoids an unprotected dividend stream. The difference is a small dividend-yield effect on the strike. |
| First tranches | Staggered first periods of 3, 6, 9 and 12 months | The only way to start a ladder at t = 0 without inventing past strike levels. |
| Narrative scenarios | Mechanism only in the core; the scenario library is built in step 2 | You asked for every number to be sourced. Shock sizes will be measured from historical series (ECB, DNB, CBS, index data) and stored with their sources, never typed in. |
| Box 3 | Deemed-return system only | Tegenbewijs (taxing actual returns) needs the actual-return rules, which are not yet sourced. The model reports an upper bound on tax. |

## 10. Study: when to convert to cash (`frontier.py`, `studies/cash_frontier.py`)

**Question.** Which cash-conversion schedule gives the best trade-off between growth and safety?
Answered by **grid search** (every candidate evaluated, nothing hidden inside an optimiser).

**Candidates (264).** Everything before the conversion phase is today's glidepath. Three things vary:

| Variable | Values | Today |
|---|---|---|
| Conversion starts (years before purchase) | 6, 5, 4, 3 | 4 |
| Cash share from start to 2 years before | 0, 10, …, 100% | 50% |
| Cash share in the last 2 years | 0, 10, …, 100%, never below the previous phase | 80% |

The non-cash remainder keeps today's composition within each phase (proportional scaling): mid phase
40% protected equity / 60% short bonds; last phase 100% short bonds. Today's glidepath is one grid point.

**Measures** (per re-calibration, 500 paths, common random numbers across all candidates):
* *growth* = P(pot ≥ own target at purchase);
* *safety A* = 5th percentile of (pot − own target), EUR: the worst 1-in-20 outcome against the goal;
* *safety B* = 5th percentile of (pot − bank-savings pot with the same contributions), EUR: the worst
  1-in-20 regret against simply saving in a bank. Added after a pilot run showed safety A is dominated by
  the target's own tail (house prices and mortgage rates), which no allocation can move much; safety B
  isolates the risk the fund controls. Both are reported.

**Frontier.** Medians across re-calibrations of growth and safety; a candidate is *dominated* if another is
≥ on both and > on one (Pareto). The non-dominated set is the frontier. Choosing one point on it needs a
stated preference (e.g. "highest growth with safety no worse than N months of saving"); the study itself
makes no such choice.

**Guarding against noise / overfitting.** The 200 bootstrap re-calibrations (option C) are split: even-numbered
for selection, odd-numbered held out. Frontier membership, medians and paired differences against today's
glidepath (median, 5–95% band, share of re-calibrations where the candidate is better) are reported on the
held-out half, separately for both regime methods. The Jaccard overlap of the two halves' frontiers measures
how stable the frontier is.

## References

* Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *JPE* 81(3).
* Merton, R. C. (1973). Theory of Rational Option Pricing. *Bell J. Econ.* 4(1).
* Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series. *Econometrica* 57(2).
* Ang, A. & Bekaert, G. (2002). International Asset Allocation with Regime Shifts. *RFS* 15(4).
* Case, K. E. & Shiller, R. J. (1989). The Efficiency of the Market for Single-Family Homes. *AER* 79(1).
* Wilson, E. B. (1927). Probable Inference, the Law of Succession, and Statistical Inference. *JASA* 22(158).
* Hahn, G. J. & Meeker, W. Q. (1991). *Statistical Intervals*. Wiley.
* Ang, A., Azimbayev, N. & Kim, A. (2026). The Self-Driving Portfolio. arXiv:2604.02279.
