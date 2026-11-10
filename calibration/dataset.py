"""Build the monthly calibration panel from the raw series.

Every transformation is declared here and in docs/CALIBRATION.md.

Model variables (NL saver's perspective):
  short_rate     EONIA - 8.5 bp up to 2019-09, then €STR (monthly mean of daily). The ECB fixed
                 EONIA = €STR + 8.5 bp from 2 Oct 2019, so subtracting 8.5 bp makes the
                 series continuous on the €STR basis.
  y2, term_spread  2y German federal zero-coupon yield (Bundesbank); spread = y2 - short_rate
  mortgage_rate  NL new mortgage loans, initial fixation > 10 years (ECB MIR, from 2003-01)
  inflation      NL HICP, annual rate
  equity         developed-market equity total return in EUR:
                 (1 + Mkt-RF + RF)_USD * (USD/EUR_{t-1} / USD/EUR_t) - 1, then log
  house          log growth of the CBS existing-homes price index (2020=100)

Regime indicators (euro area, point-in-time; publication lags applied):
  growth   real GDP y/y (quarterly; usable 2 months after quarter end) [0.6],
           minus the monthly change in unemployment (usable with 1-month lag) [0.4]
  inflation  HICP y/y [0.7]; HICP m/m with each calendar month's expanding mean
             removed (point-in-time seasonal adjustment) [0.3]
  policy   ECB deposit facility rate, month-end [0.5]; its 3-month change [0.5]
  financial conditions  CISS monthly mean [0.6]; its monthly change [0.4]
Weights are the agentic repo's MACRO_SCORING_WEIGHTS. Each US indicator is
mapped to its closest euro-area equivalent (NFP -> unemployment change,
Fed funds -> DFR, Chicago NFCI -> CISS).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sources import (
    fetch, parse_bbk, parse_eurostat, parse_cbs_house, parse_cboe, parse_ecb, parse_fred, parse_french_monthly,
)

EONIA_ESTR_SPREAD = 0.00085   # ECB press release 31 May 2019: EONIA = €STR + 8.5 bp from 2 Oct 2019
EONIA_ESTR_SOURCE = "https://www.ecb.europa.eu/press/pr/date/2019/html/ecb.pr190531~a3788de9f8.en.html"


def _months(start: str, end: str) -> np.ndarray:
    return np.arange(np.datetime64(start, "M"), np.datetime64(end, "M") + 1)


def _to_month(dates: np.ndarray) -> np.ndarray:
    return np.array([np.datetime64(str(d)[:7], "M") for d in dates])


def _monthly_mean(dates: np.ndarray, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = _to_month(dates)
    u = np.unique(m)
    return u, np.array([vals[m == k].mean() for k in u])


def _align(months: np.ndarray, m: np.ndarray, v: np.ndarray) -> np.ndarray:
    out = np.full(months.shape, np.nan)
    pos = {k: i for i, k in enumerate(months)}
    for k, x in zip(m, v):
        if k in pos:
            out[pos[k]] = x
    return out


def _step_month_end(dates: np.ndarray, vals: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Value in force at each month end for a 'date of change' series."""
    d = np.array([np.datetime64(str(x)[:10], "D") for x in dates])
    order = np.argsort(d)
    d, vals = d[order], vals[order]
    ends = (months + 1).astype("datetime64[D]") - np.timedelta64(1, "D")
    idx = np.searchsorted(d, ends, side="right") - 1
    return np.where(idx >= 0, vals[np.clip(idx, 0, None)], np.nan)


@dataclass
class Panel:
    months: np.ndarray
    v: dict[str, np.ndarray] = field(default_factory=dict)       # model variables
    ind: dict[str, np.ndarray] = field(default_factory=dict)     # regime indicators (point-in-time)
    aux: dict[str, np.ndarray] = field(default_factory=dict)     # implied-vol inputs etc.
    notes: list[str] = field(default_factory=list)


def build_panel(start: str = "1996-01", end: str | None = None) -> Panel:
    """Monthly panel. `start` is early only so indicator z-scores have history; the
    model sample starts 1999-01 (see calibrate.SAMPLE_START)."""
    get = lambda n, p: p(fetch(n)[0])                               # noqa: E731

    # ---- house prices determine the last complete month for NL
    hd, hv = get("house_nl", parse_cbs_house)
    end = end or str(_to_month(hd)[-1])
    months = _months(start, end)
    P = Panel(months)

    # ---- short rate: EONIA (basis-adjusted) until 2019-09, then €STR
    ed, ev = get("eonia", parse_ecb)
    sd, sv = get("estr", parse_ecb)
    sm, smv = _monthly_mean(sd, sv)
    eonia = _align(months, _to_month(ed), ev) / 100 - EONIA_ESTR_SPREAD
    estr = _align(months, sm, smv) / 100
    cut = np.datetime64("2019-10", "M")
    P.v["short_rate"] = np.where(months >= cut, estr, eonia)

    bd, bv = get("y2_de", parse_bbk)
    P.v["y2"] = _align(months, _to_month(bd), bv) / 100
    P.v["term_spread"] = P.v["y2"] - P.v["short_rate"]

    md, mv = get("mortgage_nl", parse_ecb)
    P.v["mortgage_rate"] = _align(months, _to_month(md), mv) / 100

    nd, nv = get("hicp_nl_anr", parse_eurostat)
    P.v["inflation"] = _align(months, _to_month(nd), nv) / 100

    # ---- equity in EUR
    fd, mkt = get("ff_dev", lambda b: parse_french_monthly(b, "Mkt-RF"))
    _, rf = get("ff_dev", lambda b: parse_french_monthly(b, "RF"))
    fm = np.array([np.datetime64(f"{s[:4]}-{s[4:6]}", "M") for s in fd])
    r_usd = _align(months, fm, (mkt + rf) / 100)
    xd, xv = get("usd_per_eur", parse_ecb)
    usd_per_eur = _align(months, _to_month(xd), xv)
    fx_ratio = np.r_[np.nan, usd_per_eur[:-1] / usd_per_eur[1:]]
    P.v["equity"] = np.log1p(r_usd) + np.log(fx_ratio)

    P.v["house_index"] = _align(months, _to_month(hd), hv)
    P.v["house"] = np.r_[np.nan, np.diff(np.log(P.v["house_index"]))]

    # ---- regime indicators (euro area), with publication lags
    gd, gv = get("gdp_ea", parse_ecb)                                # quarterly, '2024-Q2'
    q_end = np.array([np.datetime64(f"{s[:4]}-{int(s[-1]) * 3:02d}", "M") for s in gd])
    P.ind["gdp_yoy"] = _forward_fill(_align(months, q_end + 2, gv / 100))
    ud, uv = get("unemp_ea", parse_ecb)
    u = _align(months, _to_month(ud), uv)
    P.ind["neg_unemp_change"] = np.r_[np.nan, np.nan, -(u[1:-1] - u[:-2])]       # lag 1 month
    ad, av = get("hicp_ea_anr", parse_eurostat)
    P.ind["hicp_yoy"] = _align(months, _to_month(ad), av / 100)
    idd, idv = get("hicp_ea_idx", parse_eurostat)
    idx = _align(months, _to_month(idd), idv)
    P.ind["hicp_mom_sa"] = _pit_deseasonalise(np.r_[np.nan, np.diff(np.log(idx))], months)
    dd, dv = get("dfr", parse_ecb)
    dfr = _step_month_end(dd, dv / 100, months)
    dfr[months < np.datetime64("1999-01", "M")] = np.nan
    P.ind["dfr"] = dfr
    P.ind["dfr_3m_change"] = np.r_[np.full(3, np.nan), dfr[3:] - dfr[:-3]]
    cd, cv = get("ciss", parse_ecb)
    cm, cmv = _monthly_mean(cd, cv)
    ciss = _align(months, cm, cmv)
    P.ind["ciss"] = ciss
    P.ind["ciss_change"] = np.r_[np.nan, np.diff(ciss)]

    # ---- implied-volatility inputs (US proxies; see LIMITATIONS)
    vd, vv = get("vix", parse_fred)
    vm, vmv = _monthly_mean(vd, (vv / 100) ** 2)
    P.aux["vix_var"] = _align(months, vm, vmv)                      # mean implied variance in month
    P.aux["vix_min"] = np.array([np.nanmin(vv) / 100])
    ud_, mkt_d = get("ff_us_daily", lambda b: parse_french_monthly(b, "Mkt-RF"))
    _, rf_d = get("ff_us_daily", lambda b: parse_french_monthly(b, "RF"))
    dm = np.array([np.datetime64(f"{s[:4]}-{s[4:6]}", "M") for s in ud_])
    r_d = (mkt_d + rf_d) / 100
    um = np.unique(dm)
    rv = np.array([252.0 * np.mean(r_d[dm == k] ** 2) for k in um])
    P.aux["realised_var"] = _align(months, um, rv)
    kd, kv = get("skew", parse_cboe)
    km, kmv = _monthly_mean(kd, kv)
    P.aux["cboe_skew"] = _align(months, km, kmv)
    P.notes.append(f"EONIA converted to €STR basis by subtracting {EONIA_ESTR_SPREAD*1e4:.1f} bp ({EONIA_ESTR_SOURCE}).")
    return P


def _forward_fill(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    for i in range(1, len(out)):
        if np.isnan(out[i]):
            out[i] = out[i - 1]
    return out


def _pit_deseasonalise(x: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Subtract, for each month, the mean of the same calendar month over *earlier* years only."""
    cal = months.astype(int) % 12
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        prev = x[:i][(cal[:i] == cal[i]) & ~np.isnan(x[:i])]
        if len(prev) >= 3 and not np.isnan(x[i]):
            out[i] = x[i] - prev.mean()
    return out
