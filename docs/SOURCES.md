# Data sources

Each data item used by the model is listed here. The machine-readable version,
with a source URL and verification level per number, is
[`homepiggy/data/nl/policy_2026.yaml`](../homepiggy/data/nl/policy_2026.yaml).
A test fails if any policy entry lacks a source or verification level.

Verification levels:
- **primary**: read directly from the legal text or an official government page.
- **secondary**: an official explanatory page or reputable secondary source
  quoting the official figure; the primary text has not yet been read.
- **assumption**: a modelling choice, not a fact. Review it.

Retrieved 2026-11-10. Dutch rules change every 1 January.

## NL policy 2026

| Item | Value | Source | Level |
|---|---|---|---|
| Financing-cost table (97 income rows × 12 test-rate columns) | CSV, machine-extracted | [Staatscourant 2025 nr. 36471](https://zoek.officielebekendmakingen.nl/stcrt-2025-36471.pdf), Bijlage 1 Tabel 1, in force 1 Jan 2026 (Artikel II); PDF sha256 `bc3cde8f…a47ea` | primary |
| Incomes above €125,000 use the top row | n/a | none; the table ends at €125,000 | assumption |
| Income test on a 30-year annuity basis | 360 months | [Volkshuisvesting Nederland: maximale hypotheek op basis van inkomen](https://www.volkshuisvestingnederland.nl/onderwerpen/tijdelijke-regeling-hypothecair-krediet/maximale-hypotheek-op-basis-van-inkomen) | secondary |
| Test rate: < 10y fixed → max(rate, 5%) | 5% (AFM, Q2 and Q3 2026) | [Van Campen Dijkstra news](https://nieuws.vancampendijkstra.nl/related/hypotheken/maandlasten-hypotheek) | secondary |
| Singles extra loan | €17,000 if toetsinkomen > €30,000 | Threshold: [Stcrt 2025/36471](https://zoek.officielebekendmakingen.nl/stcrt-2025-36471.pdf), Art. I A. Amount: [Stcrt 2024/35962](https://zoek.officielebekendmakingen.nl/stcrt-2024-35962.odt), Art. I B (not amended for 2026) | primary |
| Student-loan gross-up factors | 1.05 … 1.40 by test rate | [Stcrt 2024/35962](https://zoek.officielebekendmakingen.nl/stcrt-2024-35962.odt), Art. I C (Art. 3a lid 2, 3) | primary |
| Energy-label extra loan amounts | €0 … €40,000 | [Stcrt 2025/36471](https://zoek.officielebekendmakingen.nl/stcrt-2025-36471.pdf), Art. I B | primary |
| Maximum LTV | 100% | [Volkshuisvesting Nederland: maximale hypotheek op basis van woningwaarde](https://www.volkshuisvestingnederland.nl/onderwerpen/tijdelijke-regeling-hypothecair-krediet/maximale-hypotheek-op-basis-van-woningwaarde-ltv) | secondary |
| Transfer tax: 2% owner-occupier, 8% otherwise | | [Rijksoverheid](https://www.rijksoverheid.nl/onderwerpen/huis-kopen/vraag-en-antwoord/vrijstelling-overdrachtsbelasting) | primary |
| Starter exemption: 18 ≤ age < 35, once only, owner-occupied, value ≤ €555,000; above the limit no exemption at all | | [Belastingdienst](https://www.belastingdienst.nl/wps/wcm/connect/bldcontentnl/belastingdienst/prive/woning/overdrachtsbelasting/startersvrijstelling), cross-checked with [Rijksoverheid](https://www.rijksoverheid.nl/onderwerpen/huis-kopen/vraag-en-antwoord/vrijstelling-overdrachtsbelasting) | primary |
| NHG limit €470,000 (€498,200 with energy measures); premium 0.4% | | [Rijksoverheid, 8 Oct 2025](https://www.rijksoverheid.nl/actueel/nieuws/2025/10/08/nhg-grens-stijgt-naar--euro--470.000-euro-afsluitpremie-blijft-04) | primary |
| Box 3: rate 36%; allowance €59,357 (€118,714 partners); deemed 1.28% bank (provisional), 6.00% other (definitive), 2.70% debts (provisional) | | [SRA, 8 Jan 2026](https://www.sra.nl/nieuws/004501/2026/01/box-3-percentages-vrijstellingen-en-bedragen-2026). Not yet cross-checked on belastingdienst.nl | secondary |

## UK (England) policy, as in force on 2026-11-23

Machine-readable: [`homepiggy/data/uk/policy_2026.yaml`](../homepiggy/data/uk/policy_2026.yaml).

| Item | Value | Source | Level |
|---|---|---|---|
| SDLT standard bands (single residential property) | 0% to £125,000; 2% to £250,000; 5% to £925,000; 10% to £1.5m; 12% above | [GOV.UK: residential property rates](https://www.gov.uk/stamp-duty-land-tax/residential-property-rates) | primary |
| SDLT first-time buyer relief | 0% to £300,000; 5% on £300,001–£500,000; no relief above £500,000 | same page | primary |
| SDLT thresholds kept at nominal values in future years | not indexed | statute fixes them in pounds | assumption |
| Maximum loan-to-value | 95% | [GOV.UK: 2025 mortgage guarantee scheme](https://www.gov.uk/government/publications/2025-mortgage-guarantee-scheme) | primary |
| Loan-to-income limit | 4.5× gross income | [FCA FG26/11](https://www.fca.org.uk/publications/finalised-guidance/fg26-11-high-loan-income-lending) (FPC flow limit: ≤15% of lending at ≥4.5×, market aggregate) | secondary; treating 4.5× as a hard limit is an assumption |
| Lifetime ISA: £4,000 a year, 25% bonus (max £1,000), open at 18–39, pay in until 50 | | [GOV.UK: Lifetime ISA](https://www.gov.uk/lifetime-isa) | primary |
| Lifetime ISA: 20% withdrawal charge for a home above £450,000; none at or below; first payment ≥ 12 months before purchase | | [GOV.UK: withdrawing from a Lifetime ISA](https://www.gov.uk/lifetime-isa/withdrawing-money-from-your-lifetime-isa) | primary |
| Lifetime ISA bonus credited one month after each payment | | not stated on GOV.UK | assumption |
| ISA allowance 2026/27 | £20,000 | [GOV.UK: Individual Savings Accounts](https://www.gov.uk/individual-savings-accounts) | primary |
| Cash ISA limit for under-65s from April 2027 | ignored | £12,000 legislated; Autumn Budget 2026 reportedly keeps £20,000; not confirmed on GOV.UK | assumption |
| Default UK saver = NL default × 0.8338 GBP/EUR | £50,000 / £335,000 / £460 a month | [ECB reference rate](https://data-api.ecb.europa.eu/service/data/EXR/M.GBP.EUR.SP00.E?format=csvdata), end October 2026 | primary (rate); rounding is a choice |

UK market data sources and hashes: [CALIBRATION_UK.md](CALIBRATION_UK.md), section 1.

## Saver inputs (user-chosen, not data)

Income, contributions, house price, horizon, student debt, DUO rate and other
buyer costs are scenario inputs the user sets. Defaults in `SaverProfile` are
illustrative only. For reference, the DUO interest rate default of 2.33% (SF35,
2026) is quoted by the
[Consumentenbond](https://www.consumentenbond.nl/hypotheek/studieschuld-hypotheek)
(secondary).

## Market data (step 2)

All 17 series (ECB, Eurostat, Deutsche Bundesbank, CBS, Kenneth French Data
Library, FRED/Cboe, Cboe) are listed with provider, series key, URL, retrieval
time and SHA-256 in [`calibration/output/manifest.json`](../calibration/output/manifest.json)
and in [CALIBRATION.md](CALIBRATION.md). Each calibrated parameter file embeds
the same list in `provenance.sources`. The ECB's fixed EONIA–€STR spread of
8.5 bp is taken from the
[ECB press release of 31 May 2019](https://www.ecb.europa.eu/press/pr/date/2019/html/ecb.pr190531~a3788de9f8.en.html).

`params/synthetic_test.yaml` remains, clearly marked, for engine tests only.
