"""Build the monthly UK calibration panel (England for house prices).

The panel has the same variables, units and indicator names as the NL panel
(calibration/dataset.py), so the classifier, the estimators and the engine are
unchanged. Every transformation is declared here and in docs/CALIBRATION_UK.md.

Model variables (UK saver's perspective, sterling):
  short_rate     SONIA, monthly average (BoE IUMASOIA, from 1997-01). SONIA was reformed in April 2018
                 (wider transaction base, volume-weighted trimmed mean); the BoE judged the level change
                 negligible, so no basis adjustment is made (declared).
  y2, term_spread  2-year nominal zero-coupon gilt yield, month end (BoE GLC curves); spread = y2 - short_rate
  mortgage_rate  Quoted rate on a 5-year fixed mortgage at 75% loan-to-value (BoE IUMBV42, from 1995-01)
  inflation      CPI all items, annual rate (ONS D7G7)
  equity         developed-market equity total return in GBP:
                 (1 + Mkt-RF + RF)_USD * (USD/GBP_{t-1} / USD/GBP_t) - 1, then log, with
                 USD/GBP = (USD per EUR) / (GBP per EUR), both ECB end-of-period reference rates
  house          log growth of the UK HPI for England, all property types (HM Land Registry, monthly from 1995)

Regime indicators (UK, point-in-time; publication lags applied):
  growth     real GDP y/y (ONS IHYR; usable 2 months after quarter end) [0.6],
             minus the change in the unemployment rate (ONS MGSX, rolling 3 months; usable with a
             2-month lag, as the Labour Force Survey is published about 6 weeks after the period) [0.4]
  inflation  CPI y/y [0.7]; CPI m/m with each calendar month's expanding mean removed [0.3]
  policy     Bank Rate, month end [0.5]; its 3-month change [0.5]
  financial conditions  UK CISS (ECB, new CISS), monthly mean [0.6]; its monthly change [0.4]
The indicator dictionary keeps the NL names (hicp_*, dfr, ciss) so the classifier is literally the same code;
the mapping is: hicp -> UK CPI, dfr -> Bank Rate, ciss -> UK CISS.
"""
from __future__ import annotations

import numpy as np

from .dataset import Panel, _align, _forward_fill, _monthly_mean, _months, _pit_deseasonalise, _step_month_end, _to_month
from .sources import (
    fetch, parse_boe, parse_cboe, parse_ecb, parse_fred, parse_french_monthly, parse_glc_2y, parse_ons, parse_ukhpi,
)

UK_SERIES = ("boe_iadb", "gilt_curve", "gbp_per_eur", "cpi_uk_anr", "cpi_uk_idx", "gdp_uk", "unemp_uk", "ciss_uk",
             "house_eng")
SHARED_SERIES = ("usd_per_eur", "ff_dev", "ff_us_daily", "vix", "skew")


def build_panel_uk(start: str = "1996-01", end: str | None = None) -> Panel:
    get = lambda n, p: p(fetch(n)[0])                               # noqa: E731

    hd, hv = get("house_eng", parse_ukhpi)
    end = end or str(_to_month(hd)[-1])
    months = _months(start, end)
    P = Panel(months)

    iadb = fetch("boe_iadb")[0]
    sd, sv = parse_boe(iadb, "IUMASOIA")
    P.v["short_rate"] = _align(months, _to_month(sd), sv / 100)
    gd2, gv2 = get("gilt_curve", parse_glc_2y)
    P.v["y2"] = _align(months, _to_month(gd2), gv2 / 100)
    P.v["term_spread"] = P.v["y2"] - P.v["short_rate"]
    md, mv = parse_boe(iadb, "IUMBV42")
    P.v["mortgage_rate"] = _align(months, _to_month(md), mv / 100)
    cd_, cv_ = get("cpi_uk_anr", parse_ons)
    P.v["inflation"] = _align(months, _to_month(cd_), cv_ / 100)

    # ---- equity in GBP
    fd, mkt = get("ff_dev", lambda b: parse_french_monthly(b, "Mkt-RF"))
    _, rf = get("ff_dev", lambda b: parse_french_monthly(b, "RF"))
    fm = np.array([np.datetime64(f"{s[:4]}-{s[4:6]}", "M") for s in fd])
    r_usd = _align(months, fm, (mkt + rf) / 100)
    ud, uv = get("usd_per_eur", parse_ecb)
    gd, gv = get("gbp_per_eur", parse_ecb)
    usd_per_gbp = _align(months, _to_month(ud), uv) / _align(months, _to_month(gd), gv)
    fx_ratio = np.r_[np.nan, usd_per_gbp[:-1] / usd_per_gbp[1:]]
    P.v["equity"] = np.log1p(r_usd) + np.log(fx_ratio)

    P.v["house_index"] = _align(months, _to_month(hd), hv)
    P.v["house"] = np.r_[np.nan, np.diff(np.log(P.v["house_index"]))]

    # ---- regime indicators (UK), with publication lags
    qd, qv = get("gdp_uk", parse_ons)                                # quarter-end month labels
    P.ind["gdp_yoy"] = _forward_fill(_align(months, _to_month(qd) + 2, qv / 100))
    nd, nv = get("unemp_uk", parse_ons)
    u = _align(months, _to_month(nd), nv)
    P.ind["neg_unemp_change"] = np.r_[np.full(3, np.nan), -(u[1:-2] - u[:-3])]     # lag 2 months
    P.ind["hicp_yoy"] = P.v["inflation"].copy()
    xd, xv = get("cpi_uk_idx", parse_ons)
    idx = _align(months, _to_month(xd), xv)
    P.ind["hicp_mom_sa"] = _pit_deseasonalise(np.r_[np.nan, np.diff(np.log(idx))], months)
    bd, bv = parse_boe(iadb, "IUDBEDR")
    br = _step_month_end(bd, bv / 100, months)
    P.ind["dfr"] = br
    P.ind["dfr_3m_change"] = np.r_[np.full(3, np.nan), br[3:] - br[:-3]]
    kd, kv = get("ciss_uk", parse_ecb)
    km, kmv = _monthly_mean(kd, kv)
    ciss = _align(months, km, kmv)
    P.ind["ciss"] = ciss
    P.ind["ciss_change"] = np.r_[np.nan, np.diff(ciss)]

    # ---- implied-volatility inputs: the same US proxies as the NL calibration (see LIMITATIONS)
    vd, vv = get("vix", parse_fred)
    vm, vmv = _monthly_mean(vd, (vv / 100) ** 2)
    P.aux["vix_var"] = _align(months, vm, vmv)
    P.aux["vix_min"] = np.array([np.nanmin(vv) / 100])
    ud_, mkt_d = get("ff_us_daily", lambda b: parse_french_monthly(b, "Mkt-RF"))
    _, rf_d = get("ff_us_daily", lambda b: parse_french_monthly(b, "RF"))
    dm = np.array([np.datetime64(f"{s[:4]}-{s[4:6]}", "M") for s in ud_])
    r_d = (mkt_d + rf_d) / 100
    um = np.unique(dm)
    rv = np.array([252.0 * np.mean(r_d[dm == k] ** 2) for k in um])
    P.aux["realised_var"] = _align(months, um, rv)
    sk_d, sk_v = get("skew", parse_cboe)
    skm, skmv = _monthly_mean(sk_d, sk_v)
    P.aux["cboe_skew"] = _align(months, skm, skmv)
    P.notes.append("SONIA used as published (no adjustment for the April 2018 reform).")
    P.notes.append("USD/GBP is the cross of the ECB USD/EUR and GBP/EUR end-of-period reference rates.")
    return P
