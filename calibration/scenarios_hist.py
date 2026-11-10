"""Historical-replay scenario library, measured from data (no typed-in shock sizes).

For each named episode the *window* is found algorithmically inside a stated
search period: the peak-to-trough of the relevant series, or the largest
12-month rise for rates. Then:
  * regime path = the primary classifier's labels for the months in the window;
  * shocks = the part of the observed move NOT explained by the calibrated
    model's expected path along that regime path (zero innovations):
      equity/house:  observed log change - sum of regime mean log growth
      rate levels:   (observed end level - deterministic AR(1) end level), scaled
                     so that the evenly spread shock, after mean reversion,
                     lands exactly on the observed end level.
Replaying an episode in the model therefore reproduces the historical
cumulative move in expectation, under the reference (full-sample) parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEVELS = ("short_rate", "term_spread", "mortgage_rate", "inflation")


@dataclass(frozen=True)
class EventRule:
    name: str
    description: str
    series: str        # which series defines the window
    rule: str          # 'drawdown' (peak->trough of an index) or 'max_rise_12m' (level)
    search_from: str
    search_to: str


EVENTS = (
    EventRule("gfc_2008", "Global financial crisis: equity peak-to-trough", "equity_index", "drawdown", "2007-01", "2009-12"),
    EventRule("euro_crisis_2011", "Euro sovereign debt crisis: equity peak-to-trough", "equity_index", "drawdown", "2011-01", "2012-12"),
    EventRule("covid_2020", "COVID-19 crash: equity peak-to-trough", "equity_index", "drawdown", "2020-01", "2020-12"),
    EventRule("rate_shock_2022", "2022 rate shock: largest 12-month rise in NL mortgage rates", "mortgage_rate", "max_rise_12m", "2021-01", "2023-12"),
    EventRule("nl_housing_slump", "NL housing slump: house-price peak-to-trough", "house_index", "drawdown", "2006-01", "2015-12"),
)


def find_window(x: np.ndarray, months: np.ndarray, rule: EventRule) -> tuple[int, int]:
    lo = int(np.searchsorted(months, np.datetime64(rule.search_from, "M")))
    hi = int(np.searchsorted(months, np.datetime64(rule.search_to, "M"), side="right")) - 1
    seg = x[lo:hi + 1]
    if rule.rule == "drawdown":
        peak = np.maximum.accumulate(seg)
        dd = seg / peak - 1.0
        b = int(np.argmin(dd))
        a = int(np.argmax(seg[: b + 1]))
        return lo + a, lo + b
    if rule.rule == "max_rise_12m":
        rise = seg[12:] - seg[:-12]
        a = int(np.nanargmax(rise))
        return lo + a, lo + a + 12
    raise ValueError(rule.rule)


def build_events(rows: dict[str, np.ndarray], labels: np.ndarray, months: np.ndarray,
                 params: dict, regimes: tuple[str, ...]) -> list[dict]:
    eq_index = np.exp(np.nancumsum(np.nan_to_num(rows["equity"])))
    house_index = np.exp(np.nancumsum(np.nan_to_num(rows["house_raw"])))
    series = {"equity_index": eq_index, "house_index": house_index, **{v: rows[v] for v in LEVELS}}
    out = []
    for ev in EVENTS:
        a, b = find_window(series[ev.series], months, ev)
        n = b - a
        if n < 1:
            continue
        path = labels[a + 1: b + 1]
        reg_names = [regimes[i] for i in path]
        mu_e = np.asarray(params["equity"]["mu"]) / 12
        mu_h = np.asarray(params["house"]["mu"]) / 12
        eq_obs = float(np.sum(rows["equity"][a + 1: b + 1]))
        h_obs = float(np.log(house_index[b] / house_index[a]))
        shocks = {"equity_log": eq_obs - float(mu_e[path].sum()),
                  "house_log": h_obs - float(mu_h[path].sum())}
        observed = {"equity_log": eq_obs, "house_log": h_obs}
        for v in LEVELS:
            hl = params[v]["half_life_months"]
            kappa = 1 - 0.5 ** (1 / hl)
            theta = np.asarray(params[v]["theta"])
            y = rows[v][a]
            if np.isnan(y) or np.isnan(rows[v][b]):
                shocks[v] = 0.0
                observed[v] = None
                continue
            for s in path:
                y = y + kappa * (theta[s] - y)
            gap = rows[v][b] - y
            decay = sum((1 - kappa) ** i for i in range(n))      # sum of decayed per-month 1/n shocks, x n
            shocks[v] = float(gap * n / decay)
            observed[v] = float(rows[v][b] - rows[v][a])
        out.append({
            "name": ev.name, "description": ev.description,
            "window": [str(months[a]), str(months[b])], "duration_months": int(n),
            "regime_path": reg_names, "shocks": shocks, "observed_change": observed,
            "detection": {"series": ev.series, "rule": ev.rule, "search": [ev.search_from, ev.search_to]},
        })
    return out
