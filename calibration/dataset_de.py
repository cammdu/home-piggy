"""Build the monthly calibration panel for Germany (house prices quarterly).

Same variables, units and indicator names as the NL panel (calibration/dataset.py). Rates and equity are
euro-area/euro series shared with NL; the rest is German. Declared in docs/CALIBRATION_DE.md.

Model variables (German saver, euros):
  short_rate     EONIA - 8.5 bp to 2019-09, then €STR (identical to NL: one euro money market)
  y2, term_spread  2-year German federal zero-coupon yield (Bundesbank); spread = y2 - short_rate (as NL)
  mortgage_rate  DE new loans for house purchase, initial fixation over 10 years (ECB MIR, from 2000-01)
  inflation      HICP Germany, annual rate (Eurostat)
  equity         developed-market equity total return in EUR (identical to NL)
  house          QUARTERLY: log change of the Destatis house price index (61262-0001) over each quarter, stored at
                 the quarter-end month; other months are NaN. The estimator treats the monthly latent growth as
                 unobserved and fits it to these quarterly changes (estimate._house_kalman_q).

Regime indicators (Germany, point-in-time; publication lags as for NL):
  growth     real GDP y/y (Eurostat namq_10_gdp; usable 2 months after quarter end) [0.6], minus the monthly
             change in the unemployment rate (Eurostat une_rt_m, SA; 1-month lag) [0.4]
  inflation  HICP Germany y/y [0.7]; HICP m/m with each calendar month's expanding mean removed [0.3]
  policy     ECB deposit facility rate, month end [0.5]; its 3-month change [0.5]
  financial conditions  German CISS (ECB new CISS), monthly mean [0.6]; its monthly change [0.4]
"""
from __future__ import annotations

import csv
import io

import numpy as np

from .dataset import (EONIA_ESTR_SPREAD, EONIA_ESTR_SOURCE, Panel, _align, _forward_fill, _monthly_mean, _months,
                      _pit_deseasonalise, _step_month_end, _to_month)
from .sources import fetch, parse_bbk, parse_cboe, parse_ecb, parse_eurostat, parse_fred, parse_french_monthly

DE_SERIES = ("house_de", "house_de_ecb", "mortgage_de", "hicp_de_anr", "hicp_de_idx", "unemp_de", "gdp_de", "ciss_de")
SHARED_SERIES_DE = ("eonia", "estr", "dfr", "y2_de", "usd_per_eur", "ff_dev", "ff_us_daily", "vix", "skew")


def parse_destatis_hpi(body: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Destatis GENESIS 61262-0001 CSV (flat 'ffcsv' or the semicolon table layout): quarter-end months, index.

    Accepts quarter labels like '1. Quartal' / 'Quartal 1' / 'Q1' / '2024-Q1' together with a year column, keeps
    the 'insgesamt' (total) series when several are present, and reads German decimal commas.
    """
    text = body.decode("utf-8-sig", "replace")
    if text.count(";") < text.count(","):
        delim = ","
    else:
        delim = ";"
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delim)]
    out = {}
    for r in rows:
        cells = [c.strip() for c in r]
        year = next((c for c in cells if len(c) == 4 and c.isdigit() and 1990 <= int(c) <= 2100), None)
        q = None
        for c in cells:
            cl = c.lower().replace(" ", "")
            for n in "1234":
                if cl in (f"{n}.quartal", f"quartal{n}", f"q{n}", f"{n}stquarter", f"{n}ndquarter", f"{n}rdquarter",
                          f"{n}thquarter", f"quarter{n}") or cl.endswith(f"-q{n}"):
                    q = int(n)
                    if cl[:4].isdigit():
                        year = cl[:4]
        if year is None or q is None:
            continue
        nums = []
        for c in cells:
            cc = c.replace(".", "").replace(",", ".") if delim == ";" else c
            try:
                x = float(cc)
            except ValueError:
                continue
            if 20.0 <= x <= 400.0 and not (len(c) == 4 and c.isdigit()):
                nums.append(x)
        if not nums:
            continue
        key = f"{year}-{q * 3:02d}"
        out.setdefault(key, nums[-1])
    if len(out) < 8:
        raise ValueError("Could not read the Destatis house price index; check the downloaded CSV layout.")
    d = np.array(sorted(out))
    return d, np.array([out[k] for k in d])


def build_panel_de(start: str = "1996-01", end: str | None = None) -> Panel:
    get = lambda n, p: p(fetch(n)[0])                               # noqa: E731

    hd, hv = get("house_de", parse_destatis_hpi)
    end = end or str(_to_month(hd)[-1])
    months = _months(start, end)
    P = Panel(months)

    ed, ev = get("eonia", parse_ecb)
    sd, sv = get("estr", parse_ecb)
    sm, smv = _monthly_mean(sd, sv)
    eonia = _align(months, _to_month(ed), ev) / 100 - EONIA_ESTR_SPREAD
    estr = _align(months, sm, smv) / 100
    P.v["short_rate"] = np.where(months >= np.datetime64("2019-10", "M"), estr, eonia)
    bd, bv = get("y2_de", parse_bbk)
    P.v["y2"] = _align(months, _to_month(bd), bv) / 100
    P.v["term_spread"] = P.v["y2"] - P.v["short_rate"]
    md, mv = get("mortgage_de", parse_ecb)
    P.v["mortgage_rate"] = _align(months, _to_month(md), mv / 100)
    nd, nv = get("hicp_de_anr", parse_eurostat)
    P.v["inflation"] = _align(months, _to_month(nd), nv / 100)

    fd, mkt = get("ff_dev", lambda b: parse_french_monthly(b, "Mkt-RF"))
    _, rf = get("ff_dev", lambda b: parse_french_monthly(b, "RF"))
    fm = np.array([np.datetime64(f"{s[:4]}-{s[4:6]}", "M") for s in fd])
    r_usd = _align(months, fm, (mkt + rf) / 100)
    xd, xv = get("usd_per_eur", parse_ecb)
    usd_per_eur = _align(months, _to_month(xd), xv)
    P.v["equity"] = np.log1p(r_usd) + np.log(np.r_[np.nan, usd_per_eur[:-1] / usd_per_eur[1:]])

    # ---- house prices: quarterly changes at quarter-end months
    hidx = _align(months, _to_month(hd), hv)
    P.v["house_index"] = hidx
    house = np.full(len(months), np.nan)
    qpos = np.where(~np.isnan(hidx))[0]
    for a, b in zip(qpos[:-1], qpos[1:]):
        if b - a == 3:
            house[b] = np.log(hidx[b] / hidx[a])
    P.v["house"] = house
    P.aux["house_quarterly"] = np.array([1.0])

    # ---- regime indicators (Germany)
    gd, gv = get("gdp_de", parse_eurostat)
    P.ind["gdp_yoy"] = _forward_fill(_align(months, _to_month(gd) + 2, gv / 100))
    ud, uv = get("unemp_de", parse_eurostat)
    u = _align(months, _to_month(ud), uv)
    P.ind["neg_unemp_change"] = np.r_[np.nan, np.nan, -(u[1:-1] - u[:-2])]       # lag 1 month (as NL)
    P.ind["hicp_yoy"] = P.v["inflation"].copy()
    idd, idv = get("hicp_de_idx", parse_eurostat)
    idx = _align(months, _to_month(idd), idv)
    P.ind["hicp_mom_sa"] = _pit_deseasonalise(np.r_[np.nan, np.diff(np.log(idx))], months)
    dd, dv = get("dfr", parse_ecb)
    dfr = _step_month_end(dd, dv / 100, months)
    dfr[months < np.datetime64("1999-01", "M")] = np.nan
    P.ind["dfr"] = dfr
    P.ind["dfr_3m_change"] = np.r_[np.full(3, np.nan), dfr[3:] - dfr[:-3]]
    cd, cv = get("ciss_de", parse_ecb)
    cm, cmv = _monthly_mean(cd, cv)
    ciss = _align(months, cm, cmv)
    P.ind["ciss"] = ciss
    P.ind["ciss_change"] = np.r_[np.nan, np.diff(ciss)]

    vd, vv = get("vix", parse_fred)
    vm, vmv = _monthly_mean(vd, (vv / 100) ** 2)
    P.aux["vix_var"] = _align(months, vm, vmv)
    P.aux["vix_min"] = np.array([np.nanmin(vv) / 100])
    ud_, mkt_d = get("ff_us_daily", lambda b: parse_french_monthly(b, "Mkt-RF"))
    _, rf_d = get("ff_us_daily", lambda b: parse_french_monthly(b, "RF"))
    dm = np.array([np.datetime64(f"{s[:4]}-{s[4:6]}", "M") for s in ud_])
    r_d = (mkt_d + rf_d) / 100
    um = np.unique(dm)
    P.aux["realised_var"] = _align(months, um, np.array([252.0 * np.mean(r_d[dm == k] ** 2) for k in um]))
    kd, kv = get("skew", parse_cboe)
    km, kmv = _monthly_mean(kd, kv)
    P.aux["cboe_skew"] = _align(months, km, kmv)
    P.notes.append(f"EONIA converted to €STR basis by subtracting {EONIA_ESTR_SPREAD*1e4:.1f} bp ({EONIA_ESTR_SOURCE}).")
    P.notes.append("House prices are quarterly (Destatis); monthly latent growth is estimated by temporal aggregation.")
    return P
