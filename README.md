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
4. **Own target (optional, default 10%)?** P(pot ≥ max(legal minimum, 10% of the
   price + buying costs)): a buffer goal the saver sets, which matters most for higher earners.

Every probability and percentile is reported as a band across 200 bootstrap
re-calibrations (parameter uncertainty), with its Monte Carlo error checked
separately.

## Status

| Step | Status |
|---|---|
| 1. Python core | done (PR #1): engine, NL rules |
| 2. Calibration from cited data, two regime methods, 200-replicate bootstrap bands, historical scenarios | done (this branch) |
| 3. Streamlit app (analysis) | planned |
| 4. Browser app (pitch) | planned |

Calibrated NL parameters live in `params/` (`nl_calibrated*.yaml`,
`nl_bootstrap*.json`, `nl_scenarios.json`). Results must be read as **bands across the bootstrap
replicates** (see `homepiggy/uncertainty.py`). Before using any number, read
**[docs/LIMITATIONS.md](docs/LIMITATIONS.md)**.

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
  uncertainty.py       option-C bands across bootstrap parameter sets
  data/nl/             sourced 2026 policy data
docs/METHODS.md        every method, equation, assumption and deviation
docs/SOURCES.md        every data source and its verification level
docs/CALIBRATION.md    estimates, validation, scenarios (generated from outputs)
docs/LIMITATIONS.md    every known limitation, with its size where measurable
calibration/           data download, regime labelling, estimation, bootstrap, validation
tools/                 reproducible extraction of official tables
params/                calibrated NL parameters, bootstrap sets, scenarios; synthetic test file
```

## Quick start

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
```

Recalibrate (downloads ~25 MB, about 10 minutes):

```bash
PYTHONPATH=. .venv/bin/python -m calibration.download
PYTHONPATH=. .venv/bin/python -m calibration.calibrate --replicates 200
PYTHONPATH=. .venv/bin/python -m calibration.band_default
PYTHONPATH=. .venv/bin/python -m calibration.write_docs
```

```python
from homepiggy.engine import RunConfig
from homepiggy.glidepath import GlidepathSpec

from homepiggy.uncertainty import load_bootstrap, compare_band

params, _ = load_bootstrap("params/nl_bootstrap.json")        # 200 re-calibrations (repo regime labels)
band = compare_band(RunConfig(market=params[0]), [GlidepathSpec(),
                    GlidepathSpec(kind="time_warp", gamma=0.6, name="late de-risk")], params, n_paths=1000)
print(band["step"]["p_reach_deposit"])      # median and 5-95% band across re-calibrations
```

## Licence

MIT. Some methods are adapted from
[chloe_agentic_architecture_for_institutional_asset_management](https://github.com/cammdu/chloe_agentic_architecture_for_institutional_asset_management)
(MIT).
