"""Registry of every raw data series used in calibration, and a cached downloader.

Every series has: provider, exact series key, URL, frequency, units and the
transformation applied. Downloads are cached under data/cache/ (gitignored,
because provider licences differ). A manifest with retrieval time and SHA-256
of each raw file is written to calibration/output/manifest.json and committed,
so every calibrated number can be traced to the exact bytes it came from.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
USER_AGENT = "home-piggy calibration (research; python-urllib)"

ECB = "https://data-api.ecb.europa.eu/service/data/{key}?format=csvdata"
BBK = "https://api.statistiken.bundesbank.de/rest/data/{key}?format=csv&lang=en"


@dataclass(frozen=True)
class Series:
    name: str
    provider: str
    key: str
    url: str
    freq: str
    units: str
    description: str
    licence_note: str


def _ecb(name: str, key: str, freq: str, units: str, desc: str) -> Series:
    return Series(name, "ECB Data Portal", key, ECB.format(key=key.replace(".", "/", 1)), freq, units, desc,
                  "ECB statistics: reuse permitted with source attribution")


EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{key}"


def _eurostat(name: str, key: str, units: str, desc: str) -> Series:
    return Series(name, "Eurostat", key, EUROSTAT.format(key=key), "M", units, desc,
                  "Eurostat: reuse permitted with source attribution (Commission Decision 2011/833/EU)")


REGISTRY: dict[str, Series] = {s.name: s for s in [
    _ecb("eonia", "FM.M.U2.EUR.4F.MM.EONIA.HSTA", "M", "% p.a., monthly average",
         "EONIA (euro overnight index average), 1999-01 to 2022-01"),
    _ecb("estr", "EST.B.EU000A2X2A25.WT", "D", "% p.a.", "Euro short-term rate (€STR), from 2019-10"),
    _ecb("dfr", "FM.D.U2.EUR.4F.KR.DFR.LEV", "D", "% p.a.", "ECB deposit facility rate (date of changes)"),
    _eurostat("hicp_nl_anr", "prc_hicp_minr?geo=NL&coicop18=TOTAL&unit=RCH_A", "% y/y",
              "HICP (ECOICOP ver.2) all items, Netherlands, annual rate of change"),
    _eurostat("hicp_ea_anr", "prc_hicp_minr?geo=EA&coicop18=TOTAL&unit=RCH_A", "% y/y",
              "HICP (ECOICOP ver.2) all items, euro area (changing composition), annual rate of change"),
    _eurostat("hicp_ea_idx", "prc_hicp_minr?geo=EA&coicop18=TOTAL&unit=I15", "index 2015=100",
              "HICP (ECOICOP ver.2) all items, euro area (changing composition), index"),
    _ecb("unemp_ea", "LFSI.M.I9.S.UNEHRT.TOTAL0.15_74.T", "M", "% of labour force",
         "Euro area unemployment rate, 15-74, seasonally adjusted"),
    _ecb("gdp_ea", "MNA.Q.Y.I9.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.GY", "Q", "% y/y",
         "Euro area real GDP, chain-linked volumes, growth on same quarter of previous year"),
    _ecb("ciss", "CISS.D.U2.Z0Z.4F.EC.SS_CIN.IDX", "D", "index 0-1",
         "Composite Indicator of Systemic Stress (CISS), euro area"),
    _ecb("usd_per_eur", "EXR.M.USD.EUR.SP00.E", "M", "USD per EUR, end of period", "ECB reference rate USD/EUR"),
    _ecb("mortgage_nl", "MIR.M.NL.B.A2C.P.R.A.2250.EUR.N", "M", "% p.a.",
         "NL bank rates, new loans to households for house purchase, initial rate fixation over 10 years"),
    Series("y2_de", "Deutsche Bundesbank", "BBSIS.M.I.ZST.ZI.EUR.S1311.B.A604.R02XX.R.A.A._Z._Z.A",
           BBK.format(key="BBSIS/M.I.ZST.ZI.EUR.S1311.B.A604.R02XX.R.A.A._Z._Z.A"), "M", "% p.a.",
           "German federal securities, 2-year residual maturity, zero-coupon yield (term structure), monthly average",
           "Bundesbank statistics: reuse permitted with source attribution"),
    Series("house_nl", "CBS StatLine", "85773NED",
           "https://opendata.cbs.nl/ODataApi/odata/85773NED/TypedDataSet", "M", "index 2020=100",
           "Bestaande koopwoningen; verkoopprijzen prijsindex 2020=100 (existing owner-occupied homes), monthly, from 1995-01",
           "CBS open data, CC BY 4.0"),
    Series("ff_dev", "Kenneth R. French Data Library", "Developed_3_Factors",
           "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_3_Factors_CSV.zip", "M",
           "% per month, USD", "Developed markets factors: Mkt-RF and RF (value-weighted developed-market equity)",
           "Free for research use; not redistributed here"),
    Series("ff_us_daily", "Kenneth R. French Data Library", "F-F_Research_Data_Factors_daily",
           "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
           "D", "% per day, USD", "US market factor, daily (for realised volatility)",
           "Free for research use; not redistributed here"),
    Series("skew", "Cboe", "SKEW",
           "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv", "D",
           "index; SKEW = 100 - 10 x (30-day risk-neutral skewness)", "Cboe SKEW index, daily",
           "Cboe data; not redistributed here"),
    Series("vix", "FRED (St. Louis Fed), source Cboe", "VIXCLS",
           "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS", "D", "% p.a. (implied vol)",
           "Cboe VIX index, daily close", "Cboe data via FRED; not redistributed here"),
]}


# ------------------------------------------------------------------ United Kingdom (England for house prices)
# The Bank of England site refuses scripted downloads, so its two files are downloaded by hand in a browser
# (user decision 2026-11-23) and placed in data/cache/ under the names below. They are then hashed and logged
# in the manifest exactly like the automatic downloads.
BOE_IADB = ("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jan/1990"
            "&Dateto=now&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N&SeriesCodes=IUMASOIA,IUDBEDR,IUMBV42")
BOE_GLC = "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/glcnominalmonthedata.zip"
MANUAL_FILES = {"boe_iadb": "boe_iadb.csv", "gilt_curve": "glcnominalmonthedata.zip"}
ONS = "https://www.ons.gov.uk/{path}/timeseries/{key}/{dataset}/data"


def _ons(name: str, key: str, dataset: str, path: str, freq: str, units: str, desc: str) -> Series:
    return Series(name, "Office for National Statistics", f"{key.upper()} ({dataset.upper()})",
                  ONS.format(path=path, key=key, dataset=dataset), freq, units, desc,
                  "ONS: Open Government Licence v3.0")


REGISTRY_UK: dict[str, Series] = {s.name: s for s in [
    Series("boe_iadb", "Bank of England Database", "IUMASOIA, IUDBEDR, IUMBV42", BOE_IADB, "M/D", "% p.a.",
           "IUMASOIA: SONIA, monthly average. IUDBEDR: official Bank Rate, daily. IUMBV42: quoted household rate, "
           "5-year fixed-rate mortgage, 75% loan-to-value, monthly. Downloaded manually in a browser.",
           "Bank of England statistics: reuse permitted with source attribution"),
    Series("gilt_curve", "Bank of England (yield curves)", "GLC nominal month-end spot curve, 2-year", BOE_GLC,
           "M", "% p.a.", "UK government nominal zero-coupon (spot) yield, 2-year maturity, month end "
           "(sheet '4. spot curve'). Downloaded manually in a browser.",
           "Bank of England statistics: reuse permitted with source attribution"),
    _ecb("gbp_per_eur", "EXR.M.GBP.EUR.SP00.E", "M", "GBP per EUR, end of period", "ECB reference rate GBP/EUR"),
    _ons("cpi_uk_anr", "d7g7", "mm23", "economy/inflationandpriceindices", "M", "% y/y",
         "CPI all items, annual rate (D7G7)"),
    _ons("cpi_uk_idx", "d7bt", "mm23", "economy/inflationandpriceindices", "M", "index 2015=100",
         "CPI all items index (D7BT)"),
    _ons("gdp_uk", "ihyr", "qna", "economy/grossdomesticproductgdp", "Q", "% y/y",
         "Real GDP, growth on the same quarter a year earlier, CVM SA (IHYR)"),
    _ons("unemp_uk", "mgsx", "lms", "employmentandlabourmarket/peoplenotinwork/unemployment", "M", "% of labour force",
         "Unemployment rate, aged 16+, seasonally adjusted, rolling 3 months (MGSX)"),
    _ecb("ciss_uk", "CISS.D.GB.Z0Z.4F.EC.SS_CIN.IDX", "D", "index 0-1",
         "New Composite Indicator of Systemic Stress (CISS), United Kingdom, daily"),
    Series("house_eng", "HM Land Registry (UK House Price Index)", "UKHPI England, all property types",
           "https://landregistry.data.gov.uk/app/ukhpi/download/new.csv?from=1995-01-01&to=2030-12-01"
           "&location=http%3A%2F%2Flandregistry.data.gov.uk%2Fid%2Fregion%2Fengland"
           "&thm%5B%5D=property_type&thm%5B%5D=buyer_status&in%5B%5D=hpi&in%5B%5D=avg",
           "M", "index Jan 2023=100", "UK HPI, England, all property types, monthly from 1995-01",
           "Open Government Licence v3.0 (contains HM Land Registry data, Crown copyright)"),
]}
# ------------------------------------------------------------------ Germany
# The Destatis house price index (GENESIS table 61262-0001, quarterly from 2000) needs an account for scripted
# access, so it is downloaded by hand in a browser (user decision 2026-11-24) and saved as data/cache/destatis_61262.csv.
# The ECB's copy of the same Destatis index (from 2005) is registered as a cross-check.
DESTATIS_HPI = "https://www-genesis.destatis.de/datenbank/online/statistic/61262/table/61262-0001"
MANUAL_FILES["house_de"] = "destatis_61262.csv"

REGISTRY_DE: dict[str, Series] = {s.name: s for s in [
    Series("house_de", "Destatis (GENESIS-Online)", "61262-0002 Häuserpreisindex, Deutschland, Quartale",
           DESTATIS_HPI, "Q", "index", "House price index for owner-occupied residential property (existing and new), "
           "Germany, quarterly from 2000. Downloaded manually in a browser.", "Destatis: Datenlizenz Deutschland 2.0"),
    _ecb("house_de_ecb", "RESR.Q.DE._T.N._TR.TVAL.4D0.TB.N.IX", "Q", "index",
         "House price index, total, Germany (Destatis Häuserpreisindex, via ECB), quarterly from 2005 Q1"),
    _ecb("mortgage_de", "MIR.M.DE.B.A2C.P.R.A.2250.EUR.N", "M", "% p.a.",
         "DE bank rates, new loans to households for house purchase, initial rate fixation over 10 years"),
    _eurostat("hicp_de_anr", "prc_hicp_minr?geo=DE&coicop18=TOTAL&unit=RCH_A", "% y/y",
              "HICP (ECOICOP ver.2) all items, Germany, annual rate of change"),
    _eurostat("hicp_de_idx", "prc_hicp_minr?geo=DE&coicop18=TOTAL&unit=I15", "index 2015=100",
              "HICP (ECOICOP ver.2) all items, Germany, index"),
    Series("unemp_de", "Eurostat", "une_rt_m?geo=DE&s_adj=SA&age=TOTAL&sex=T&unit=PC_ACT",
           EUROSTAT.format(key="une_rt_m?geo=DE&s_adj=SA&age=TOTAL&sex=T&unit=PC_ACT"), "M", "% of labour force",
           "Unemployment rate, Germany, total, seasonally adjusted, monthly",
           "Eurostat: reuse permitted with source attribution (Commission Decision 2011/833/EU)"),
    Series("gdp_de", "Eurostat", "namq_10_gdp?geo=DE&s_adj=SCA&unit=CLV_PCH_SM&na_item=B1GQ",
           EUROSTAT.format(key="namq_10_gdp?geo=DE&s_adj=SCA&unit=CLV_PCH_SM&na_item=B1GQ"), "Q", "% y/y",
           "Real GDP, Germany, chain-linked volumes, change on the same quarter of the previous year",
           "Eurostat: reuse permitted with source attribution (Commission Decision 2011/833/EU)"),
    _ecb("ciss_de", "CISS.D.DE.Z0Z.4F.EC.SS_CIN.IDX", "D", "index 0-1",
         "New Composite Indicator of Systemic Stress (CISS), Germany, daily"),
]}
REGISTRY_ALL: dict[str, Series] = {**REGISTRY, **REGISTRY_UK, **REGISTRY_DE}


def _get(url: str, attempts: int = 6, timeout: int = 120) -> bytes:
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            if body.lstrip()[:15].lower().startswith((b"<html", b"<!doctype")) and "cbs" not in url:
                raise RuntimeError("provider returned an HTML error page")
            return body
        except (urllib.error.URLError, TimeoutError, RuntimeError, ConnectionError) as exc:
            last = exc
            time.sleep(min(10 * (i + 1), 60))     # polite back-off
    raise RuntimeError(f"Download failed after {attempts} attempts: {url} ({last})")


def fetch(name: str, refresh: bool = False) -> tuple[bytes, dict]:
    """Return raw bytes for a registered series (cached) and its manifest entry."""
    s = REGISTRY_ALL[name]
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.raw"
    meta_path = CACHE / f"{name}.meta.json"
    if path.exists() and meta_path.exists() and not refresh:
        return path.read_bytes(), json.loads(meta_path.read_text())
    if name in MANUAL_FILES:
        src = CACHE / MANUAL_FILES[name]
        if not src.exists():
            raise FileNotFoundError(f"{name}: download {s.url} in a browser and save it as {src}.")
        body = src.read_bytes()
        meta = {**asdict(s), "retrieved_utc": datetime.fromtimestamp(src.stat().st_mtime, timezone.utc).isoformat(
                    timespec="seconds"), "retrieval": "manual browser download",
                "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
        path.write_bytes(body)
        meta_path.write_text(json.dumps(meta, indent=2))
        return body, meta
    if s.provider == "CBS StatLine":
        body = _get_cbs(s.url)
    else:
        body = _get(s.url)
    path.write_bytes(body)
    meta = {**asdict(s), "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
    meta_path.write_text(json.dumps(meta, indent=2))
    return body, meta


def _get_cbs(url: str) -> bytes:
    """CBS OData pages results 10,000 rows at a time; follow odata.nextLink and concatenate."""
    rows, nxt = [], url + "?$format=json"
    while nxt:
        page = json.loads(_get(nxt))
        rows.extend(page["value"])
        nxt = page.get("odata.nextLink")
    return json.dumps({"value": rows}).encode()


# ---------------------------------------------------------------- parsers -> (dates, values)
def _month_index(s: str) -> np.datetime64:
    return np.datetime64(s[:7], "M")


def parse_ecb(body: bytes) -> tuple[np.ndarray, np.ndarray]:
    import csv
    rd = csv.DictReader(io.StringIO(body.decode("utf-8")))
    d, v = [], []
    for row in rd:
        if row.get("OBS_VALUE", "") in ("", "NaN"):
            continue
        d.append(row["TIME_PERIOD"])
        v.append(float(row["OBS_VALUE"]))
    if not d:
        raise ValueError("ECB response contained no observations.")
    return np.array(d), np.array(v)


def parse_bbk(body: bytes) -> tuple[np.ndarray, np.ndarray]:
    text = body.decode("utf-8", "replace").splitlines()
    d, v = [], []
    for line in text:
        parts = line.split(",")
        if len(parts) >= 2 and len(parts[0]) == 7 and parts[0][:4].isdigit() and parts[0][4] == "-":
            try:
                v.append(float(parts[1]))
                d.append(parts[0])
            except ValueError:
                continue
    if not d:
        raise ValueError("Bundesbank response contained no observations.")
    return np.array(d), np.array(v)


def parse_french_monthly(body: bytes, column: str) -> tuple[np.ndarray, np.ndarray]:
    """First (monthly) block of a French-library CSV inside a zip. Returns YYYY-MM dates and % values."""
    z = zipfile.ZipFile(io.BytesIO(body))
    text = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    header_i = next(i for i, l in enumerate(text) if l.replace(" ", "").startswith(",Mkt-RF"))
    cols = [c.strip() for c in text[header_i].split(",")]
    j = cols.index(column)
    d, v = [], []
    for line in text[header_i + 1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(cols) or not (parts[0].isdigit() and len(parts[0]) in (6, 8)):
            break
        d.append(parts[0])
        v.append(float(parts[j]))
    return np.array(d), np.array(v)


def parse_fred(body: bytes) -> tuple[np.ndarray, np.ndarray]:
    lines = body.decode("utf-8").splitlines()[1:]
    d, v = [], []
    for l in lines:
        a, b = l.split(",")
        if b not in (".", ""):
            d.append(a)
            v.append(float(b))
    return np.array(d), np.array(v)


def parse_cbs_house(body: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Monthly national price index from CBS 85773NED (periods like '2024MM05')."""
    rows = json.loads(body)["value"]
    out = {}
    for r in rows:
        p = r.get("Perioden", "")
        if "MM" not in p:
            continue
        val = None
        for k in ("PrijsindexVerkoopprijzen_1",):
            if r.get(k) is not None:
                val = r[k]
                break
        if val is None:
            continue
        out[f"{p[:4]}-{p[6:8]}"] = float(val)
    if not out:
        raise ValueError("CBS response contained no monthly index values.")
    d = np.array(sorted(out))
    return d, np.array([out[k] for k in d])


def parse_cboe(body: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Cboe history CSV: DATE (MM/DD/YYYY), VALUE -> ISO dates."""
    d, v = [], []
    for line in body.decode("utf-8").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 2 or not parts[1].strip():
            continue
        m, dd, y = parts[0].split("/")
        d.append(f"{y}-{int(m):02d}-{int(dd):02d}")
        v.append(float(parts[1]))
    return np.array(d), np.array(v)


def parse_eurostat(body: bytes) -> tuple[np.ndarray, np.ndarray]:
    """JSON-stat 2.0 response with a single non-time series -> YYYY-MM dates and values."""
    d = json.loads(body)
    if "error" in d:
        raise ValueError(f"Eurostat error: {d['error']}")
    t = d["dimension"]["time"]["category"]["index"]
    inv = {v: k for k, v in t.items()}
    sizes = d["size"]
    if np.prod(sizes) != len(t):
        raise ValueError("Eurostat query returned more than one series; tighten the filter.")
    pairs = sorted((_quarter_end(inv[int(k)]), float(v)) for k, v in d["value"].items())
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def _quarter_end(label: str) -> str:
    """'2024-Q2' -> '2024-06' (quarter-end month); other labels unchanged."""
    if len(label) == 7 and label[4:6] == "-Q":
        return f"{label[:4]}-{int(label[-1]) * 3:02d}"
    return label


# ---------------------------------------------------------------- UK parsers
def parse_boe(body: bytes, code: str) -> tuple[np.ndarray, np.ndarray]:
    """Bank of England IADB CSV, one or more series ('DATE,CODE1,CODE2..'; 'DD Mon YYYY'); one column out."""
    import csv
    from datetime import datetime as _dt
    rows = list(csv.reader(io.StringIO(body.decode("utf-8-sig", "replace"))))
    j = [c.strip() for c in rows[0]].index(code)
    d, v = [], []
    for r in rows[1:]:
        if len(r) > j and r[j].strip():
            d.append(_dt.strptime(r[0].strip(), "%d %b %Y").date().isoformat())
            v.append(float(r[j]))
    if not d:
        raise ValueError(f"Bank of England file contained no observations for {code}.")
    return np.array(d), np.array(v)


_MON = {m: i + 1 for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"))}


def parse_ons(body: bytes) -> tuple[np.ndarray, np.ndarray]:
    """ONS time-series JSON: monthly observations if present, else quarterly ('YYYY Qn' -> quarter-end month)."""
    j = json.loads(body)
    if j.get("months"):
        rows = [(f"{r['year']}-{_MON[r['month'][:3].upper()]:02d}", r["value"]) for r in j["months"]]
    elif j.get("quarters"):
        rows = [(f"{r['year']}-{int(r['quarter'][-1]) * 3:02d}", r["value"]) for r in j["quarters"]]
    else:
        raise ValueError("ONS response has neither monthly nor quarterly observations.")
    rows = [(a, float(b)) for a, b in rows if str(b).strip() not in ("", "..")]
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


def parse_ukhpi(body: bytes, column: str = "House price index All property types") -> tuple[np.ndarray, np.ndarray]:
    import csv
    rows = list(csv.reader(io.StringIO(body.decode("utf-8"))))
    h = rows[0]
    j, p = h.index(column), h.index("Period")
    d, v = [], []
    for r in rows[1:]:
        if len(r) > j and r[j].strip():
            d.append(r[p][:7])
            v.append(float(r[j]))
    if not d:
        raise ValueError("UK HPI response contained no observations.")
    return np.array(d), np.array(v)


def _xlsx_sheet(raw: bytes, sheet_name: str) -> list[dict[str, str]]:
    """Minimal .xlsx reader (standard library only): rows of {column letter: cell text} for one sheet."""
    import re
    import xml.etree.ElementTree as ET
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
          "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    x = zipfile.ZipFile(io.BytesIO(raw))
    wb = ET.fromstring(x.read("xl/workbook.xml"))
    rid = next(s.get(f"{{{ns['r']}}}id") for s in wb.find("m:sheets", ns) if s.get("name") == sheet_name)
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(x.read("xl/_rels/workbook.xml.rels"))}
    path = "xl/" + rels[rid].lstrip("/").replace("xl/", "", 1)
    shared = []
    if "xl/sharedStrings.xml" in x.namelist():
        shared = ["".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t"))
                  for si in ET.fromstring(x.read("xl/sharedStrings.xml"))]
    out = []
    for row in ET.fromstring(x.read(path)).find("m:sheetData", ns):
        cells = {}
        for c in row:
            v = c.find("m:v", ns)
            if v is None:
                continue
            cells[re.match(r"[A-Z]+", c.get("r")).group()] = shared[int(v.text)] if c.get("t") == "s" else v.text
        out.append(cells)
    return out


def parse_glc_2y(body: bytes) -> tuple[np.ndarray, np.ndarray]:
    """BoE GLC nominal month-end archive (zip of .xlsx): 2-year spot yield from sheet '4. spot curve'.
    The 'years:' row holds maturities; column A holds Excel date serials (1900 system)."""
    z = zipfile.ZipFile(io.BytesIO(body))
    out = {}
    for name in sorted(z.namelist()):
        rows = _xlsx_sheet(z.read(name), "4. spot curve")
        years = next(r for r in rows if r.get("A", "").strip().lower().startswith("years"))
        col = next(k for k, v in years.items() if k != "A" and abs(float(v) - 2.0) < 1e-6)
        for r in rows:
            try:
                serial = float(r.get("A", ""))
            except ValueError:
                continue
            if col not in r:
                continue
            day = np.datetime64("1899-12-30") + np.timedelta64(int(serial), "D")
            out[str(day)[:7]] = float(r[col])
    d = np.array(sorted(out))
    return d, np.array([out[k] for k in d])


def parse_ecb_quarterly(body: bytes) -> tuple[np.ndarray, np.ndarray]:
    d, v = parse_ecb(body)
    return np.array([_quarter_end(x) for x in d]), v
