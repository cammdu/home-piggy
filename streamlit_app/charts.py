"""Altair chart builders. Form choices (dataviz method):
  * interval estimates -> dot (median) + rule (5-95% band), one row per strategy
  * distributions over time -> fan (nested quantile areas, single hue) + median line
  * composition over time -> stacked area of mean weights
  * signed correlations -> diverging horizontal bars around zero
  * regime history -> categorical strip (heat row) over months
Every chart has tooltips; every view also has a table below it in the app.
"""
from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd

from .presets import DIVERGING, SEQ_BLUE, SERIES, SLEEVE_NAMES

AXIS = {"labelColor": "#52514e", "titleColor": "#52514e", "gridColor": "#e7e6e2", "domainColor": "#c9c8c3"}


def _strategy_scale(names: list[str]) -> alt.Scale:
    return alt.Scale(domain=names, range=SERIES[: len(names)])


def interval_chart(df: pd.DataFrame, value_fmt: str = ".0%", title: str = "") -> alt.Chart:
    """df columns: strategy, method, median, p05, p95. Facets by method when >1."""
    names = list(dict.fromkeys(df["strategy"]))
    base = alt.Chart(df).encode(
        y=alt.Y("strategy:N", sort=names, title=None, axis=alt.Axis(**AXIS, labelLimit=220)),
        color=alt.Color("strategy:N", scale=_strategy_scale(names), legend=None),
        tooltip=[alt.Tooltip("strategy:N"), alt.Tooltip("method:N"),
                 alt.Tooltip("median:Q", format=value_fmt, title="median"),
                 alt.Tooltip("p05:Q", format=value_fmt, title="5th pct"),
                 alt.Tooltip("p95:Q", format=value_fmt, title="95th pct")],
    )
    rule = base.mark_rule(strokeWidth=2).encode(
        x=alt.X("p05:Q", title=title, axis=alt.Axis(format=value_fmt, tickCount=5, **AXIS)), x2="p95:Q")
    dot = base.mark_point(filled=True, size=90, stroke="#fcfcfb", strokeWidth=2).encode(x="median:Q")
    label = base.mark_text(align="left", dx=8, dy=-10, fontSize=11, color="#52514e").encode(
        x="median:Q", text=alt.Text("median:Q", format=value_fmt))
    layer = alt.layer(rule, dot, label).properties(height=max(60, 34 * len(names)))
    if df["method"].nunique() > 1:
        order = sorted(df["method"].unique(), key=lambda m: (not m.startswith("Repo"), m))
        return layer.facet(column=alt.Column("method:N", title=None, sort=order)).resolve_scale(x="shared")
    return layer


def fan_chart(detail: dict[str, Any], label: str, real: bool = False) -> alt.Chart:
    """Pot fan (5-95, 25-75, median) vs the saver's target (median + 5-95 dashed)."""
    d = detail[label]
    pot = d["pot_real" if real else "pot"]
    years = [m / 12 for m in d["months"]]
    df = pd.DataFrame({"year": years, **{f"pot_{k}": v for k, v in pot.items()},
                       **{f"tgt_{k}": v for k, v in d["own_target"].items()}})
    x = alt.X("year:Q", title="Years from start", axis=alt.Axis(**AXIS))
    eur = alt.Axis(format="~s", title="EUR (nominal)" if not real else "EUR (2026 prices)", **AXIS)
    top = max(df["pot_p95"].max(), 0 if real else df["tgt_p95"].max()) * 1.05
    ysc = alt.Scale(domain=[0, float(top)], nice=True, zero=True)
    outer = alt.Chart(df).mark_area(color=SEQ_BLUE["200"], opacity=0.55).encode(
        x=x, y=alt.Y("pot_p05:Q", axis=eur, scale=ysc), y2="pot_p95:Q")
    inner = alt.Chart(df).mark_area(color=SEQ_BLUE["300"], opacity=0.75).encode(x=x, y="pot_p25:Q", y2="pot_p75:Q")
    med = alt.Chart(df).mark_line(color=SEQ_BLUE["600"], strokeWidth=2).encode(x=x, y="pot_p50:Q")
    layers = [outer, inner, med]
    if not real:
        tgt = alt.Chart(df).mark_line(color=SERIES[1], strokeWidth=2).encode(x=x, y="tgt_p50:Q")
        tlo = alt.Chart(df).mark_line(color=SERIES[1], strokeWidth=1, strokeDash=[4, 3]).encode(x=x, y="tgt_p05:Q")
        thi = alt.Chart(df).mark_line(color=SERIES[1], strokeWidth=1, strokeDash=[4, 3]).encode(x=x, y="tgt_p95:Q")
        layers += [tgt, tlo, thi]
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["year"], empty=False)
    tip_fields = [alt.Tooltip("year:Q", format=".1f", title="year")] + [
        alt.Tooltip(f"pot_{k}:Q", format=",.0f", title=f"pot {k}") for k in ("p05", "p25", "p50", "p75", "p95")]
    if not real:
        tip_fields += [alt.Tooltip("tgt_p50:Q", format=",.0f", title="target median")]
    rule = alt.Chart(df).mark_rule(color="#8a8984").encode(x=x, opacity=alt.condition(nearest, alt.value(0.6), alt.value(0)),
                                                             tooltip=tip_fields).add_params(nearest)
    return alt.layer(*layers, rule).properties(height=340)


def allocation_chart(detail: dict[str, Any], label: str) -> alt.Chart:
    d = detail[label]
    rows = []
    for m, w in zip(d["months"], d["weights_mean"]):
        for name, x in zip(SLEEVE_NAMES, w):
            rows.append({"year": m / 12, "sleeve": name, "weight": x})
    df = pd.DataFrame(rows)
    return alt.Chart(df).mark_area(stroke="#fcfcfb", strokeWidth=1).encode(
        x=alt.X("year:Q", title="Years from start", axis=alt.Axis(**AXIS)),
        y=alt.Y("weight:Q", stack="normalize", title="Average allocation", axis=alt.Axis(format="%", **AXIS)),
        color=alt.Color("sleeve:N", scale=alt.Scale(domain=SLEEVE_NAMES, range=SERIES[:4]),
                        sort=SLEEVE_NAMES, legend=alt.Legend(orient="top", title=None)),
        order=alt.Order("sleeve_order:Q"),
        tooltip=[alt.Tooltip("year:Q", format=".1f"), "sleeve:N", alt.Tooltip("weight:Q", format=".0%")],
    ).transform_calculate(sleeve_order=f"indexof({SLEEVE_NAMES!r}, datum.sleeve)").properties(height=260)


def drivers_chart(drivers: dict[str, float]) -> alt.Chart:
    df = pd.DataFrame({"parameter": list(drivers), "rho": list(drivers.values())})
    return alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=14).encode(
        y=alt.Y("parameter:N", sort=alt.EncodingSortField("rho", op="sum", order="ascending"), title=None,
                axis=alt.Axis(**AXIS, labelLimit=300)),
        x=alt.X("rho:Q", title="Spearman rank correlation with the outcome", scale=alt.Scale(domain=[-1, 1]),
                axis=alt.Axis(**AXIS)),
        color=alt.condition("datum.rho >= 0", alt.value(DIVERGING["pos"]), alt.value(DIVERGING["neg"])),
        tooltip=["parameter:N", alt.Tooltip("rho:Q", format="+.2f")],
    ).properties(height=max(80, 26 * len(df)))


def regime_strip(df: pd.DataFrame, regimes: list[str]) -> alt.Chart:
    """df: month (datetime), method, regime."""
    return alt.Chart(df).mark_rect().encode(
        x=alt.X("yearmonth(month):T", title=None, axis=alt.Axis(**AXIS, format="%Y")),
        y=alt.Y("method:N", title=None, axis=alt.Axis(**AXIS)),
        color=alt.Color("regime:N", scale=alt.Scale(domain=regimes, range=SERIES[:4]), sort=regimes,
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("yearmonth(month):T", title="month"), "method:N", "regime:N"],
    ).properties(height=110)
