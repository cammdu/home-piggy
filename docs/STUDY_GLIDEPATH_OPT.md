# Study: optimising the glidepath (equity → protected equity → short bonds → cash)

Method: [METHODS.md §11](METHODS.md). Results: `studies/output/glidepath_opt.json` (raw `.npz` per method).
Saver: €60k income, €400k house today, €550/month, 10 years, deposit target 10% of the price + buying costs
(income test off). 106 relay glidepaths + the original, 200 bootstrap re-calibrations × 500 paths, both regime methods.

## Result

**Chosen glidepath:** shares until 4 years before buying; then 75% buffered shares / 25% shares until 2 years;
then 75% short bonds / 25% buffered shares until purchase. No separate cash phase.

Held-out half of the re-calibrations (not used for choosing):

| | Growth (median pot ÷ target) | Safety (CVaR₁₀% of pot ÷ target) | Safer than bank in | All three tests | Reach target | With income limit |
|---|---|---|---|---|---|---|
| **Optimised**, repo | **1.72** | **0.83** | 92% | **60%** | 89% | 31% |
| Original, repo | 1.53 | 0.77 | 97% | 42% | 84% | 29% |
| Savings account, repo | 1.37 | 0.70 | – | – | 77% | 27% |
| **Optimised**, HMM | **1.76** | 0.99 | 82% | **66%** | 96% | 17% |
| Original, HMM | 1.54 | 0.99 | 88% | 45% | 95% | 15% |
| Savings account, HMM | 1.33 | 0.91 | – | – | 91% | 13% |

* The optimised glidepath grows more than the original in 97% (repo) / 96% (HMM) of held-out re-calibrations.
* Crisis replays ending at the purchase date (5 episodes × 2 methods, 40 re-calibrations each): it stays at
  least as safe as a savings account in **10 of 10**, and grows more than the original in all 10.

## What to keep in mind

* **The safety bar did not bind.** All 106 candidates were at least as safe as a savings account, so the choice
  was decided by growth. Under HMM the optimised glidepath's safety equals the original's (0.99), not better.
* **The winner sits at the edge of the grid** (latest equity handover, latest bond handover, largest blend).
  Later or softer handovers were not tested and might do better.
* **No cash phase.** In this model short bonds beat cash over the last two years with little extra tail risk.
  This depends on the bond model (constant-duration 1–3y fund; LIMITATIONS D/J).
* **Measure 2 (income limit) barely moves** (29% → 31% repo): the income limit, not the fund, decides it.
