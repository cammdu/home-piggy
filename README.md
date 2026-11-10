# home-piggy

A glidepath fund model for first-time home buyers aged about 20, who invest
8–10 years before buying. The fund moves gradually from **equity →
protected equity (put-spread buffer) → short-dated bonds → cash** as the
purchase date approaches.

The Netherlands comes first; the UK and Germany are planned. The model is
**numbers only**: every result comes from deterministic code and sourced
data, with no LLM or discretionary judgement in any calculation.

## What it answers

For a saver and a glidepath, across thousands of simulated futures:

1. **Reach the deposit?** P(pot ≥ NL affordability gap at the target date).
   The gap is price + buyer costs − maximum mortgage under the 2026 Dutch
   lending rules.
2. **Outpace cash?** P(pot > the same contributions in a savings account),
   on the same paths.
3. **Keep up with housing?** P(money-weighted return ≥ house-price growth).

Every probability and percentile is reported with its 95% Monte Carlo
interval.

## Status

| Step | Status |
|---|---|
| 1. Python core (this branch) | done: engine, NL rules, 84 tests |
| 2. Calibration from cited data, plus the historical scenario library | next |
| 3. Streamlit app (analysis) | planned |
| 4. Browser app (pitch) | planned |

Market parameters are **not calibrated yet**. `params/synthetic_test.yaml` is
synthetic and only for tests. Don't use its outputs.

## Layout

```
homepiggy/
  regimes.py           Markov regimes + forced narrative paths
  market.py            regime-switching simulator (equity, house, rates, mortgage, inflation)
  options.py           Black-Scholes with regime-dependent smile
  sleeves.py           equity, put-spread buffer ladder, 1-3y bonds, cash
  glidepath.py         step / linear / time-warp / target-vol / lock-in
  portfolio.py         monthly accounting, contributions, rebalancing, tax
  affordability_nl.py  NL max mortgage + affordability gap (2026 rules)
  tax_nl.py            box 3 (2026)
  metrics.py           Wilson / order-statistic intervals, IRR, drawdown
  engine.py            run(), compare() on common random numbers
  data/nl/             sourced 2026 policy data
docs/METHODS.md        every method, equation, assumption and deviation
docs/SOURCES.md        every data source and its verification level
tools/                 reproducible extraction of official tables
params/                parameter files (synthetic test file only, for now)
```

## Quick start

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
```

```python
from homepiggy.config import load_market_params
from homepiggy.engine import RunConfig, compare
from homepiggy.glidepath import GlidepathSpec

mp = load_market_params("params/synthetic_test.yaml")   # synthetic: testing only
res = compare(RunConfig(market=mp, n_paths=5000),
              [GlidepathSpec(), GlidepathSpec(kind="time_warp", gamma=0.6, name="late de-risk")])
print(res["step"].summary["p_reach_deposit"])
```

## Licence

MIT. Some methods are adapted from
[chloe_agentic_architecture_for_institutional_asset_management](https://github.com/cammdu/chloe_agentic_architecture_for_institutional_asset_management)
(MIT).
