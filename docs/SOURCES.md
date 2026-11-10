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

## Saver inputs (user-chosen, not data)

Income, contributions, house price, horizon, student debt, DUO rate and other
buyer costs are scenario inputs the user sets. Defaults in `SaverProfile` are
illustrative only. For reference, the DUO interest rate default of 2.33% (SF35,
2026) is quoted by the
[Consumentenbond](https://www.consumentenbond.nl/hypotheek/studieschuld-hypotheek)
(secondary).

## Market parameters

**None are in the repository yet.** `params/synthetic_test.yaml` is
explicitly marked synthetic and exists only to test the engine. The loader
refuses any parameter file that is not marked synthetic and has no
`provenance.sources`. Step 2 (calibration) will estimate them from cited
series, for example ECB €STR/EONIA and yield curves, DNB mortgage rates, CBS
house price index (PBK), Eurostat HICP, MSCI World EUR / Fama-French global
factors, and VSTOXX. Each file will record its series IDs and retrieval dates.
