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
    s = REGISTRY[name]
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.raw"
    meta_path = CACHE / f"{name}.meta.json"
    if path.exists() and meta_path.exists() and not refresh:
        return path.read_bytes(), json.loads(meta_path.read_text())
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
    pairs = sorted((inv[int(k)], float(v)) for k, v in d["value"].items())
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
