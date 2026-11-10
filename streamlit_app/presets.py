"""Strategy presets and small helpers for the Streamlit app (no Streamlit imports here, so it is testable)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from homepiggy.glidepath import DEFAULT_TABLE, GlidepathSpec

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "params"

# Categorical palette (dataviz reference instance, light mode), fixed order, validated:
# adjacent CVD dE >= 9.1, normal-vision dE >= 19.6. Three slots are < 3:1 contrast -> charts carry
# direct labels / tooltips and every chart has a table view.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = {"200": "#9ec5f4", "300": "#6da7ec", "450": "#2a78d6", "600": "#184f95"}
DIVERGING = {"neg": "#e34948", "pos": "#2a78d6", "mid": "#f0efec"}
SLEEVE_NAMES = ["Equity", "Protected equity", "Short bonds", "Cash"]

PRESETS: dict[str, dict[str, Any]] = {
    "Agreed step glidepath": {"kind": "step"},
    "Linear glide": {"kind": "linear"},
    "Late de-risk (γ = 0.6)": {"kind": "time_warp", "gamma": 0.6},
    "Early de-risk (γ = 1.5)": {"kind": "time_warp", "gamma": 1.5},
    "Target volatility (12% → 1%)": {"kind": "target_vol", "vol_start": 0.12, "vol_end": 0.01},
    "Lock in when target reached": {"kind": "lock_in", "lock_multiple": 1.0},
    "All cash": {"kind": "step", "table": [{"from_years": 15, "to_years": 0, "weights": [0, 0, 0, 1]}]},
}

METHODS = {"Repo classifier (primary)": "", "HMM (cross-check)": "_hmm"}


def default_table_rows() -> list[dict[str, float]]:
    return [{"from_years": r["from_years"], "to_years": r["to_years"],
             **{n: 100.0 * w for n, w in zip(SLEEVE_NAMES, r["weights"])}} for r in DEFAULT_TABLE]


def table_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Editable-table rows (weights in %) -> engine table (fractions). Validation happens in the engine."""
    out = []
    for r in rows:
        w = [float(r[n]) / 100.0 for n in SLEEVE_NAMES]
        out.append({"from_years": float(r["from_years"]), "to_years": float(r["to_years"]), "weights": w})
    return out


def build_glidepaths(selected: list[str], custom_rows: list[dict[str, Any]] | None) -> list[GlidepathSpec]:
    gps = []
    for name in selected:
        if name == "Custom table":
            gps.append(GlidepathSpec(kind="step", table=table_from_rows(custom_rows or default_table_rows()),
                                     name=name))
        else:
            gps.append(GlidepathSpec(name=name, **PRESETS[name]))
    return gps


def load_events(method_suffix: str) -> list[dict[str, Any]]:
    data = json.loads((PARAMS / "nl_scenarios.json").read_text())
    key = "hmm" if method_suffix == "_hmm" else "repo_classifier"
    return data["events"][key]
