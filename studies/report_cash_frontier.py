"""Render the cash-frontier study: SVG charts (docs/figures/) and tables (docs/STUDY_CASH_FRONTIER_TABLES.md).

All numbers come from studies/output/cash_frontier_<method>.json; plotted values are medians on the
HELD-OUT half of the re-calibrations, while frontier membership is decided on the selection half.
Usage: PYTHONPATH=. python studies/report_cash_frontier.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "studies" / "output"
FIG = ROOT / "docs" / "figures"
METHODS = {"repo": "Repo classifier", "hmm": "HMM"}
SAFETY = {"bad_case": "Bad case vs own target (EUR, 5th pct)",
          "bad_vs_bank": "Bad case vs bank savings (EUR, 5th pct)"}
# dataviz reference palette: frontier = series 1 blue, today = series 2 orange, others = muted ink
BLUE, ORANGE, GREY, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#c9c8c3", "#2b2a27", "#52514e", "#e7e6e2", "#fcfcfb"


def _eur(x: float) -> str:
    return f"{'−' if x < 0 else ''}€{abs(x)/1000:,.1f}k"


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    import math
    span = hi - lo or 1.0
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = math.ceil(lo / step) * step
    return [start + i * step for i in range(int((hi - start) / step) + 1)]


def svg_frontier(rows: list[dict], safety: str, title: str) -> str:
    W, H, L, R, T, B = 640, 400, 70, 20, 44, 52
    xs = [r["bad_case_test"] for r in rows]; ys = [r["growth_test"] for r in rows]
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
    px = (x1 - x0) * 0.06 or 1; py = (y1 - y0) * 0.08 or 0.01
    x0, x1, y0, y1 = x0 - px, x1 + px, max(0, y0 - py), min(1, y1 + py)
    X = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)  # noqa: E731
    Y = lambda v: H - B - (v - y0) / (y1 - y0) * (H - T - B)  # noqa: E731
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="11">',
         f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
         f'<text x="{L}" y="18" font-size="13" font-weight="600" fill="{INK}">{title}</text>',
         f'<g font-size="11" fill="{MUTED}"><circle cx="{L+4}" cy="31" r="4" fill="{BLUE}"/><text x="{L+12}" y="35">frontier (chosen on other half)</text>'
         f'<circle cx="{L+214}" cy="31" r="5" fill="{ORANGE}" stroke="{SURF}" stroke-width="2"/><text x="{L+223}" y="35">today (start 4y, 50% → 80%)</text>'
         f'<circle cx="{L+404}" cy="31" r="3" fill="{GREY}"/><text x="{L+411}" y="35">other candidates</text></g>']
    for t in _ticks(y0, y1):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="{GRID}"/>'
                 f'<text x="{L-6}" y="{Y(t)+4:.1f}" text-anchor="end" fill="{MUTED}">{t:.1%}</text>')
    for t in _ticks(x0, x1):
        o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{T}" y2="{H-B}" stroke="{GRID}"/>'
                 f'<text x="{X(t):.1f}" y="{H-B+16}" text-anchor="middle" fill="{MUTED}">{_eur(t)}</text>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-12}" text-anchor="middle" fill="{MUTED}">{SAFETY[safety]} — safer →</text>')
    o.append(f'<text transform="translate(16 {(T+H-B)/2}) rotate(-90)" text-anchor="middle" fill="{MUTED}">P(reach own target) — more growth →</text>')
    for r in rows:
        if not r["frontier_selection"] and not r["is_baseline"]:
            o.append(f'<circle cx="{X(r["bad_case_test"]):.1f}" cy="{Y(r["growth_test"]):.1f}" r="3" fill="{GREY}" opacity="0.7"><title>{r["label"]}</title></circle>')
    fr = sorted([r for r in rows if r["frontier_selection"]], key=lambda r: r["bad_case_test"])
    for r in fr:
        o.append(f'<circle cx="{X(r["bad_case_test"]):.1f}" cy="{Y(r["growth_test"]):.1f}" r="4.5" fill="{BLUE}" stroke="{SURF}" stroke-width="2"><title>{r["label"]}</title></circle>')
    mid_x = (L + W - R) / 2
    for r in (fr[0], fr[-1]) if fr else ():
        right = X(r["bad_case_test"]) > mid_x
        anchor, dx = ("end", -8) if right else ("start", 8)
        o.append(f'<text x="{X(r["bad_case_test"])+dx:.1f}" y="{Y(r["growth_test"])-8:.1f}" text-anchor="{anchor}" fill="{INK}">{r["label"]}</text>')
    b = [r for r in rows if r["is_baseline"]][0]
    o.append(f'<circle cx="{X(b["bad_case_test"]):.1f}" cy="{Y(b["growth_test"]):.1f}" r="6" fill="{ORANGE}" stroke="{SURF}" stroke-width="2"><title>{b["label"]} (today)</title></circle>')
    o.append(f'<text x="{X(b["bad_case_test"]):.1f}" y="{Y(b["growth_test"])+18:.1f}" text-anchor="middle" fill="{INK}" font-weight="600">today</text>')
    o.append("</svg>")
    return "\n".join(o)


def table(a: dict) -> str:
    rows = a["rows"]
    fr = sorted([r for r in rows if r["frontier_selection"] or r["is_baseline"]], key=lambda r: -r["growth_test"])
    h = ("| Candidate | On held-out frontier | P(reach target) | Δ vs today, pp [5–95%] | Bad case | "
         "Δ vs today [5–95%] | Better on both (share of held-out re-cal.) |\n|---|---|---|---|---|---|---|\n")
    lines = []
    for r in fr:
        name = r["label"] + (" **(today)**" if r["is_baseline"] else "")
        lines.append(
            f"| {name} | {'yes' if r['frontier_test'] else 'no'} | {r['growth_test']:.1%} | "
            f"{100*r['dg_median_test']:+.1f} [{100*r['dg_p05_test']:+.1f}, {100*r['dg_p95_test']:+.1f}] | "
            f"{_eur(r['bad_case_test'])} | {_eur(r['ds_median_test'])} [{_eur(r['ds_p05_test'])}, {_eur(r['ds_p95_test'])}] | "
            f"{r['share_better_both_test']:.0%} |")
    return h + "\n".join(lines)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    md = ["# Cash-conversion frontier: tables (generated)\n",
          "Generated by `studies/report_cash_frontier.py` from `studies/output/`. Values are medians on the "
          "held-out half of the re-calibrations; the frontier was chosen on the other half. Method: METHODS.md §10.\n"]
    for m, mname in METHODS.items():
        f = OUT / f"cash_frontier_{m}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        s = d["setup"]
        md.append(f"\n## {mname}\n\n{d['bad_case']['n_candidates']} candidates × {d['bad_case']['n_replicates']} "
                  f"re-calibrations × {s['n_paths']} paths; params `{s['params_file']}` (sha256 {s['params_sha256'][:12]}…), "
                  f"run {s['run_date']}, {s['runtime_s']/60:.0f} min.\n")
        for k in SAFETY:
            a = d[k]
            # the chart's x-axis always reads the chosen safety metric
            rows = [dict(r) for r in a["rows"]]
            (FIG / f"cash_frontier_{m}_{k}.svg").write_text(svg_frontier(rows, k, f"{mname}: growth vs {SAFETY[k].split(' (')[0].lower()}"))
            md.append(f"\n### Safety = {SAFETY[k]}\n\n![frontier](figures/cash_frontier_{m}_{k}.svg)\n\n"
                      f"Frontier size selection/held-out: {a['n_frontier_selection']}/{a['n_frontier_test']}; "
                      f"overlap (Jaccard) {a['frontier_overlap_jaccard']:.2f}; today on frontier: "
                      f"selection {a['baseline_on_frontier']['selection']}, held-out {a['baseline_on_frontier']['test']}.\n\n"
                      + table(a) + "\n")
    (ROOT / "docs" / "STUDY_CASH_FRONTIER_TABLES.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
