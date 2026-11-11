# Limitations

Everything known to be imperfect, simplified or uncertain in home-piggy (steps 1 and 2), and how big it
is where that can be measured. Read this before using any output, especially in a pitch.
Severity: **High** = can change a headline conclusion; **Medium** = shifts numbers noticeably;
**Low** = minor or conservative.

Measured values below come from `calibration/output/report.json` (calibration of 2026-11-10);
[CALIBRATION.md](CALIBRATION.md) has the full tables.

---

## A. Data

| # | Limitation | Severity | Size / evidence | Mitigation |
|---|---|---|---|---|
| A1 | **Short history.** Regime-conditional estimates use 296 months (2002-02 to 2026-09), not 1999 onward: the classifier needs 24 months of history and the euro-area unemployment series starts 2000-03. That is roughly two full credit cycles. | High | Late-cycle has only 25 months under the repo classifier; bootstrap 5–95% ranges for regime means are wide (e.g. Recession equity mean roughly −14% to +5% p.a.). | Option-C bands show this directly; never quote a single number. |
| A2 | **Mortgage rates only from 2003-01** (ECB MIR series start). Correlations involving mortgage rates use 2003+ complete cases. | Low | 11 fewer months. | None needed. |
| A3 | **Equity is a USD developed-market index converted to EUR**, not MSCI World EUR (licensed). The Fama-French developed index excludes emerging markets. | Low | Composition differs slightly from a typical retail global tracker. | Swap in a licensed index if available. |
| A4 | **Option-pricing inputs are US proxies.** The volatility risk premium uses VIX vs US realised volatility; skew uses Cboe SKEW (S&P 500 options). No free euro-area option-surface history was available. | Medium | vrp 1.11, skew 0.13 per unit log-moneyness. The European premium and skew may differ. | Re-estimate with VSTOXX / Euro Stoxx 50 surface data if licensed. |
| A5 | **Skew conversion is an approximation.** SKEW → 1-year smile slope uses the Gram-Charlier expansion of Backus, Foresi & Wu (2004) and iid scaling of skewness (1/√horizon). Empirically, skew decays more slowly with maturity. | Medium | Probably *understates* 1-year put skew, so the cost of the 90% short put is understated and the net put-spread cost slightly overstated. Direction is roughly neutral to mildly conservative. | Sensitivity on `implied_vol.skew`. |
| A6 | **Data revisions / no real-time vintages.** GDP and unemployment are final revised values (with publication lags applied), not the values known at the time. | Low | Affects historical labels, not forward simulation. | Use real-time vintages (ECB RTD) if needed. |
| A7 | **The CBS index is a national average of existing homes.** First-time buyers buy cheaper and often in different regions, where growth can differ substantially (e.g. Randstad vs periphery). | Medium | Not measured. | Regional CBS tables (85792NED, COROP) could be added. |
| A8 | **HICP switched to ECOICOP v2 (Eurostat, 2026).** The ECB's legacy HICP series stop at 2025-12, so all HICP data is taken from Eurostat `prc_hicp_minr` for consistency. | Low | Small level differences vs legacy series. | None needed. |

## B. Regimes

| # | Limitation | Severity | Size / evidence | Mitigation |
|---|---|---|---|---|
| B1 | **The two regime methods disagree.** The repo classifier and the HMM give the same label in 38.5% of months (Cohen's κ = 0.21). They also disagree about today: repo says Recession (48%), HMM says Late-cycle (≈100%). | High | See CALIBRATION §2 and the band comparison in §6. | Results shown under both; any conclusion should hold under both. |
| B2 | **The repo classifier has no memory.** It labels each month independently, so regimes switch every 2–5 months on average. That is much faster than business cycles, and it weakens the regime structure (regime means average out quickly). | High | Expected spells: 4.2 / 5.0 / 2.3 / 1.9 months (repo) vs 10.6 / 29.5 / 18.4 / 13.0 (HMM). | HMM variant as alternative. |
| B3 | **The repo's thresholds are placeholders.** The repo itself marks the threshold matrix "UNSPECIFIED IN MANUSCRIPT". Its US indicators were mapped to euro-area equivalents (NFP → unemployment change, Fed funds → ECB deposit rate, NFCI → CISS), which is a judgement about equivalence. | Medium | Not measurable without an independent "true" regime history. | HMM cross-check is fully data-driven. |
| B4 | **HMM labels are ex post (smoothed, full sample)** and its states are *named* by overlap with the repo labels. The names are economically plausible (CALIBRATION §2) but the HMM finds its own states. | Medium | HMM "Expansion" is a low-inflation, easy-policy state (2014–16), not a classic boom. | Read HMM regime names with care. |
| B5 | **Labelling uncertainty is not bootstrapped.** Labels are held fixed and resampled with the data; the classification itself is not re-run per replicate. | Medium | Bands understate total uncertainty. | Comparing the two methods partly covers this. |
| B6 | **Time-homogeneous transitions.** P does not depend on how long a regime has lasted or on the economy's state. | Low | Standard simplification. | None. |
| B7 | **Absorbing regimes in some HMM bootstrap replicates.** HMM spells are long, so a resampled history can contain a regime that is never observed ending, and that regime becomes permanent within the replicate. | Medium | 20 of 200 HMM replicates (0 of 200 repo). These widen the HMM bands' extremes. | Driver analysis uses 10-year occupancy, which is always defined. Kept rather than dropped, so the bands aren't trimmed after the fact. |

## C. Estimation

| # | Limitation | Severity | Size / evidence | Mitigation |
|---|---|---|---|---|
| C1 | **Rates are near-unit-root over the sample.** Without constraints, regime target levels were absurd (35% short rate, −12%). Each regime's long-run level is now held inside the observed sample range (constrained LS). | Medium | Binding for: short rate Late-cycle (repo), mortgage Late-cycle and Recovery. Half-life bootstrap range for the short rate is wide (repo ≈30–190 months). | Constraint is explicit and reported. Rates can never be projected beyond 2002–2026 extremes in the long run (shocks still can). |
| C2 | **House-price persistence is bounded at φ ≤ 0.99** (the engine's stationarity bound). In some bootstrap replicates the likelihood pushes φ to the bound. | Low | 4.0% of repo replicates, 0.5% of HMM replicates. | Explicit bound; a near-unit-root and φ = 0.99 are indistinguishable in 24 years of data. |
| C10 | **House-price mean is set by method of moments, not likelihood.** With φ near 1 the maximum-likelihood level was biased: the model's average growth came out at 6.0% (repo) / 5.5% (HMM) p.a. against a sample mean of 3.7%, about 20–25% too-high prices after 10 years. The frequency-weighted regime mean is now constrained to equal the sample mean; regime differences and dynamics are still estimated by likelihood. | Medium | Constraint holds exactly (tested). | The long-run house-price drift is the 2002–2026 average (3.7% p.a.). If the future differs (see F2), so will results. |
| C3 | **House-price likelihood has multiple optima.** Bootstrap φ ranges from about 0.2–0.4 (5th pct) to 0.96 (95th pct). Some resampled histories favour a short-memory explanation. | Medium | Bands include both. | Full-sample fit uses 3 starts; bootstrap warm-starts from the full-sample optimum. |
| C4 | **Gaussian innovations; real ones are fat-tailed.** Excess kurtosis of residuals: mortgage ≈ 15, inflation ≈ 15, short rate ≈ 7 (ECB rate steps). | Medium | One-month checks: short-rate KS p ≈ 0.000, 12.5% outside the 90% interval (repo). Inflation and mortgage KS p ≤ 0.001. | Mixing across regimes and the data-derived historical shocks add tails. Within-regime tails are still too thin. |
| C5 | **Correlations shrink to (near) zero.** Monthly residual correlations are small (max |ρ| ≈ 0.25) and fat-tailed, so Ledoit-Wolf shrinks them heavily, often fully. Co-movement comes almost entirely through shared regimes and mean reversion, not same-month shocks. | Medium | LW intensity 1.0 in most regimes. | Data-driven; historical scenarios add joint shocks. |
| C6 | **Fallbacks for thin regimes.** When a bootstrap replicate has fewer than 12 months of a regime, that regime's parameters revert to full-sample values, which narrows the band for that regime. | Medium | Repo method: some fallback in 24.5% of replicates (almost all Late-cycle); HMM: 4.0%. | Reported; HMM variant has larger regime samples. |
| C7 | **Block length chosen by the most persistent series** (house prices, ~22 months, Politis-White). For fast series (equity, inflation) the blocks are longer than needed, which is conservative. | Low | Block lengths per series in CALIBRATION §3. | None needed. |
| C8 | **Mortgage-rate moves lean slightly high; 10-year coverage of mortgage changes is below nominal.** Historical 10-year mortgage-rate changes sit on average at the 31st–33rd percentile of simulated changes, and 71% (repo) / 86% (HMM) fall inside the 5–95% range (nominal 90%). House growth is now centred (41st–43rd percentile, 100% coverage after C10). | Medium | In-sample, overlapping windows. | Treat high-mortgage-rate outcomes with some caution. |
| C9 | **Validation is in-sample.** No out-of-sample backtest; the 10-year windows overlap heavily (30 windows over 296 months). | Medium | Coverage statistics are descriptive, not formal tests. | A rolling out-of-sample exercise is possible but needs more history. |

## D. Model structure (engine)

| # | Limitation | Severity | Notes |
|---|---|---|---|
| D1 | **Only regime-level co-movement of house prices and mortgage rates.** No direct affordability-feedback channel (higher rates → lower prices beyond what regimes capture). | Medium | Typical of reduced-form models. |
| D2 | **Bond sleeve ignores convexity and credit**, and uses the German 2y yield for a euro government 1–3y fund. | Low | Error < 0.25 bp/month. |
| D3 | **Options on the total-return index (q = 0)**; real buffer products are usually on price indices. | Low | Small dividend effect on strikes. |
| D4 | **Black-Scholes pricing with a deterministic smile.** No stochastic volatility or jumps; implied volatility moves only with regime. | Medium | Protection cost doesn't spike within a regime the way it did in March 2020. |
| D5 | **Option trading cost 0.20% per reset and TERs are design assumptions**, not market quotes. | Medium | Parameters in `ProductParams`; vary them. |
| D6 | **Incomes grow deterministically** (CPI × real growth). No unemployment or income shocks. | Medium | Overstates certainty about borrowing capacity. |
| D7 | **The saver always contributes** and never withdraws early. | Low | Behavioural risk not modelled. |
| D8 | **Lock-in and target-vol strategies assume the regime is observed.** | Low | Real-time regime detection is noisier (see B1). |
| D9 | **Narrative replays reproduce historical moves in expectation under the reference parameters.** Under other bootstrap parameter sets the same shock gives a slightly different total move. | Low | Shocks are residuals relative to the reference set. |

## E. NL rules and tax

| # | Limitation | Severity | Notes |
|---|---|---|---|
| E1 | **2026 lending rules are frozen in real terms** (CPI-indexed thresholds, house-index-indexed price limits). Future changes (e.g. the announced 2027 starter limit) are not modelled. | Medium | Rules change every 1 January. |
| E2 | **Box 3: deemed-return system only.** The *tegenbewijsregeling* (taxing actual returns if lower) and the planned actual-return system are not modelled. | Low | Tax is an upper bound. |
| E3 | **Box 3 timing:** anniversaries stand in for 1 January, and tax is paid immediately. | Low | Slightly conservative. |
| E4 | **Some rules are only secondarily verified:** 5% test rate, 30-year annuity basis, 100% LTV, box 3 figures (see SOURCES.md). | Low | Primary text not yet read. |
| E5 | **Incomes above €125,000 use the top table row** (assumption: the table ends there). | Low | Rarely relevant for first-time buyers. |
| E6 | **NHG eligibility is tested on the loan, not total costs.** | Low | Simplification. |
| E7 | **Single-buyer focus.** Couples (two incomes, partner allowance) are supported only through `single=False` and a combined income; the second-earner rules are not modelled in detail. | Medium | |
| E8 | **The own savings target (default 10% of price + costs) is a chosen goal, not a rule.** It doesn't model the lower mortgage rates Dutch banks offer at lower loan-to-value, so the benefit of the buffer isn't priced in. | Low | Rate tiers by loan-to-value would need sourcing from bank rate sheets. |

## F. Uncertainty method (option C)

| # | Limitation | Severity | Notes |
|---|---|---|---|
| F1 | **Bands cover parameter uncertainty given the model and the labels, not model uncertainty.** The difference between the two regime methods (B1) is model uncertainty and is shown separately. | High | Read the two methods' bands together. |
| F2 | **The bootstrap assumes the 2002–2026 history is representative.** Regimes or shocks absent from the sample (e.g. 1970s-style persistent inflation, a pre-2002 rate environment) cannot appear in the parameter bands. | High | Historical replays and custom shocks can stress beyond the sample. |
| F3 | **200 replicates** give 5th/95th percentiles from about the 10th lowest/highest value. The band edges themselves carry sampling error of a few percentage points. | Low | Increase `--replicates` for publication. |
| F4 | **Monte Carlo noise within each replicate** is checked via `mc_noise_ratio`. Above 0.25, increase `n_paths`. | Low | Reported per outcome. |

## H. Streamlit app

| # | Limitation | Severity | Notes |
|---|---|---|---|
| H1 | **Default runs use 50 of the 200 re-calibrations** (evenly spaced) and 500 paths, for speed (about 15–30 s per method). The 5th/95th band edges then rest on 2–3 replicates each, so they are noisier than in CALIBRATION.md. | Medium | Set re-calibrations to 200 and paths to 1,000+ before quoting numbers; the MC noise ratio warns when paths are too few. |
| H2 | **Path charts (fan, allocation, at-purchase table) show one parameter set**: the median re-calibration by P(reach own target) of the first strategy. They aren't bands, and mixing paths across re-calibrations was rejected (option B). | Low | Labelled in the app. |
| H3 | **Driver analysis uses rank correlations** between replicate parameters and outcomes. It is descriptive (parameters are correlated with each other), not causal. | Low | |
| H4 | **"If you bought today" uses the latest observed mortgage rate** (ECB MIR, NL, fixed > 10 years) for every fixed period. Shorter fixed periods carry different rates, though the 5% test-rate floor still applies. | Low | |

## G. Engineering and reproducibility

| # | Limitation | Severity | Notes |
|---|---|---|---|
| G1 | **Raw data is not committed** (licences differ). Re-downloading later can return revised data; hashes in `manifest.json` reveal any change. | Low | |
| G2 | **Corporate network blocks PyPI** on the development machine; packages had to be installed on another network. | Low | Environment issue, not model. |
| G3 | **Editable install ignored on this Mac** (hidden `.pth` files); use `PYTHONPATH=.` or pytest's `pythonpath`. | Low | Environment issue. |
