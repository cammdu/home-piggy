# Study: optimised glidepath for the UK (England)

_Run 2026-11-24, `studies/optimize_glidepath.py --country uk`; raw results `studies/output/glidepath_opt_uk.json`. Same candidates, measures and pre-agreed selection rule as the NL study ([STUDY_GLIDEPATH_OPT.md](STUDY_GLIDEPATH_OPT.md), METHODS §11); UK data, rules and wrappers as in METHODS §12._

**Saver:** the NL default in pounds (£50,000 income, £335,000 home, £460 a month, £834 to start, age 20, 10 years, deposit 10% + stamp duty, income test off), Lifetime ISA on. 500 paths per re-calibration, 200 re-calibrations per regime method (runtime 565.5 + 383.3 min on a heavily loaded machine).

**Chosen glidepath:** PE 4y | bonds 2y | no cash | blend 25% — global equity until 4 years before buying; 75% protected equity / 25% global equity until 2 years; 75% short bonds / 25% protected equity until purchase. **The same glidepath as the NL study.**

## Held-out results (odd-numbered re-calibrations; medians)

| Method | Strategy | Growth | Safety (CVaR₁₀%) | P(reach target) | P(all three) | P(income-limit gap) | safer than bank | grows more than original |
|---|---|---|---|---|---|---|---|---|
| repo | optimised | 1.39 | 0.73 | 81% | 67% | 22% | 99% | 98% |
| repo | original | 1.21 | 0.68 | 71% | 45% | 20% | – | – |
| repo | savings account | 1.05 | 0.60 | 55% | – | 18% | – | – |
| hmm | optimised | 1.36 | 0.62 | 74% | 59% | 25% | 98% | 99% |
| hmm | original | 1.20 | 0.57 | 67% | 46% | 23% | – | – |
| hmm | savings account | 1.04 | 0.51 | 53% | – | 20% | – | – |

## Crisis replays (episode ends on the purchase date; 40 re-calibrations each)

| Method | Episode | window | winner safety − bank | winner growth | original growth |
|---|---|---|---|---|---|
| repo | gfc_2008 | 2007-10 → 2009-02 | +0.121 | 2.11 | 1.99 |
| repo | euro_crisis_2011 | 2011-06 → 2011-09 | +0.092 | 1.34 | 1.21 |
| repo | covid_2020 | 2020-01 → 2020-03 | +0.094 | 1.34 | 1.21 |
| repo | rate_shock_2022 | 2021-10 → 2022-10 | +0.075 | 1.23 | 1.12 |
| repo | uk_housing_slump | 2007-09 → 2009-03 | +0.141 | 2.22 | 2.03 |
| hmm | gfc_2008 | 2007-10 → 2009-02 | +0.100 | 1.86 | 1.75 |
| hmm | euro_crisis_2011 | 2011-06 → 2011-09 | +0.081 | 1.31 | 1.19 |
| hmm | covid_2020 | 2020-01 → 2020-03 | +0.083 | 1.32 | 1.19 |
| hmm | rate_shock_2022 | 2021-10 → 2022-10 | +0.059 | 1.20 | 1.10 |
| hmm | uk_housing_slump | 2007-09 → 2009-03 | +0.120 | 1.94 | 1.77 |

## Reading the result

- All 106 candidates passed the safety bar under both methods, so growth decided, as in NL. Unlike NL's HMM case, the winner is also safer than the original glidepath on the held-out half under both methods (repo 0.73 vs 0.68; HMM 0.62 vs 0.57).
- The winner sits at the edge of the grid (latest handovers, largest blend), as in NL; later or softer handovers were not tested.
- The next five candidates by growth are all late-handover, 25%-blend variants within 0.03 of the winner's growth, so the conclusion 'stay in equity late, hand over gradually' is robust; the exact year is less certain.
- Outcomes are lower than in NL. Two UK features push that way: stamp duty relief disappears above £500,000, and the calibrated house-price growth for England is higher than for the Netherlands (e.g. 5.7% vs 4.4% a year in Expansion), so the deposit target rises faster. Their separate contributions were not measured.
