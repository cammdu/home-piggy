from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from homepiggy.config import load_yaml, market_params_from_dict
from homepiggy.market import MarketPaths

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "params" / "synthetic_test.yaml"


@pytest.fixture(scope="session")
def synth_dict() -> dict:
    return load_yaml(SYNTH)["market"]


@pytest.fixture()
def market(synth_dict):
    return market_params_from_dict(synth_dict)


def deterministic_dict(synth_dict: dict, rate: float = 0.03, infl: float = 0.0) -> dict:
    """All volatilities zero, flat constant rates: outcomes have closed forms."""
    d = copy.deepcopy(synth_dict)
    k = len(d["regimes"])
    for g in ("equity", "house"):
        d[g]["sigma"] = [0.0] * k
    d["house"]["ar1"] = 0.0
    for v, lvl in (("short_rate", rate), ("term_spread", 0.0), ("mortgage_rate", 0.04), ("inflation", infl)):
        d[v]["sigma"] = [0.0] * k
        d[v]["theta"] = [lvl] * k
        d[v]["initial"] = lvl
    return d


def hand_market(equity_index: np.ndarray, rate: float = 0.0, regime: int = 0) -> MarketPaths:
    """MarketPaths with a hand-specified equity index (P, T+1) and flat rates."""
    eq = np.asarray(equity_index, float)
    P, T1 = eq.shape
    z = np.zeros((P, T1))
    return MarketPaths(
        regimes=np.full((P, T1), regime, dtype=np.int8),
        equity_logret=np.diff(np.log(eq), axis=1),
        equity_index=eq,
        house_index=np.ones((P, T1)),
        short_rate=z + rate,
        term_spread=z.copy(),
        mortgage_rate=z + 0.04,
        inflation=z.copy(),
        cpi=np.ones((P, T1)),
    )
