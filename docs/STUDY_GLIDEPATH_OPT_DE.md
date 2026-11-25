# Study: optimised glidepath for Germany

_Run 2026-11-24, `studies/optimize_glidepath.py --country de`; raw results `studies/output/glidepath_opt_de.json`. Same candidates, measures and pre-agreed selection rule as the NL study ([STUDY_GLIDEPATH_OPT.md](STUDY_GLIDEPATH_OPT.md), METHODS §11); German data, rules and fund tax as in METHODS §13._

**Saver:** the NL default (€60,000 income, €400,000 home, €550 a month, €1,000 to start, age 20, 10 years, own target 10% of the price + buying costs, income test off), Bavaria (3.5% transfer tax). The FirstHome fund is taxed as one German investment fund (advance lump sum each year, partial exemption by equity share, deemed sale on a change of category, exit tax at purchase); the savings account pays tax on its interest every year. 500 paths per re-calibration, 200 re-calibrations per regime method (runtime 71.6 + 59.7 min).

**Chosen glidepath:** PE 4y | bonds 2y | no cash | blend 25%: global equity until 4 years before buying; 75% protected equity / 25% global equity until 2 years; 75% short bonds / 25% protected equity until purchase. **The same glidepath as the NL and UK studies.**

## Held-out results (odd-numbered re-calibrations; medians)

| Method | Strategy | Growth | Safety (CVaR₁₀%) | P(reach target) | P(all three) | P(income-limit gap) | safer than bank | grows more than original |
|---|---|---|---|---|---|---|---|---|
| repo | optimised | 1.40 | 0.91 | 89% | 81% | 23% | 95% | 100% |
| repo | original | 1.21 | 0.83 | 80% | 65% | 18% | 99% | – |
| repo | savings account | 1.02 | 0.73 | 53% | – | 15% | – | – |
| hmm | optimised | 1.34 | 0.86 | 88% | 74% | 30% | 83% | 94% |
| hmm | original | 1.19 | 0.83 | 79% | 56% | 24% | 90% | – |
| hmm | savings account | 1.03 | 0.75 | 55% | – | 18% | – | – |

The selection half gives the same picture (repo: growth 1.34 vs 1.17 original; HMM: 1.34 vs 1.17).

## Crisis replays (episode ends on the purchase date; 40 re-calibrations each)

| Method | Episode | window | winner safety − bank | winner growth | original growth | winner safety | original safety |
|---|---|---|---|---|---|---|---|
| repo | gfc_2008 | 2007-05 → 2009-02 | +0.101 | 1.37 | 1.31 | 0.84 | 0.85 |
| repo | euro_crisis_2011 | 2011-02 → 2011-09 | +0.164 | 1.37 | 1.23 | 0.86 | 0.80 |
| repo | covid_2020 | 2020-01 → 2020-03 | +0.158 | 1.38 | 1.25 | 0.87 | 0.81 |
| repo | rate_shock_2022 | 2021-11 → 2022-11 | +0.155 | 1.39 | 1.24 | 0.87 | 0.81 |
| repo | de_housing_slump | 2022-06 → 2024-03 | +0.198 | 1.78 | 1.58 | 1.11 | 1.02 |
| hmm | gfc_2008 | 2007-05 → 2009-02 | +0.013 | 1.30 | 1.30 | 0.82 | 0.89 |
| hmm | euro_crisis_2011 | 2011-02 → 2011-09 | +0.078 | 1.31 | 1.22 | 0.83 | 0.81 |
| hmm | covid_2020 | 2020-01 → 2020-03 | +0.074 | 1.33 | 1.23 | 0.82 | 0.82 |
| hmm | rate_shock_2022 | 2021-11 → 2022-11 | +0.064 | 1.30 | 1.22 | 0.82 | 0.82 |
| hmm | de_housing_slump | 2022-06 → 2024-03 | +0.108 | 1.57 | 1.45 | 1.02 | 1.00 |

## Reading the result

- All 106 candidates passed the safety bar under both methods, so growth decided, as in NL and UK.
- On the held-out half the winner is safer than the original glidepath under both methods (repo 0.91 vs 0.83; HMM 0.86 vs 0.83).
- **Weak spot: a 2008-style crash ending at purchase, under the HMM.** The winner then grows no more than the original (1.30 vs 1.30), is less safe (0.82 vs 0.89), and is only just safer than the savings account (+0.013). The reason is that it holds equity later. Under the repo method the same replay costs a little safety (0.84 vs 0.85).
- The winner sits at the edge of the grid (latest handovers, largest blend), as in NL and UK; later or softer handovers were not tested.
- The next six candidates by growth (selection half, repo) are all bonds-at-2-years, 25%-blend variants within 0.031 of the winner's growth, so 'stay in equity late, hand over gradually' is robust; the exact year is less certain.
- Growth is lower than in NL (held-out repo 1.40 vs 1.72). The German target (10% of the price plus costs, since a 100%-LTV loan needs only the costs in cash) and the yearly fund tax differ from NL; their separate contributions were not measured.
- The German sample is short (2005–2026) and the regime methods agree on only 51% of months (κ 0.24), so the HMM rows carry more weight than usual as a check (LIMITATIONS L1–L4 for Germany).
