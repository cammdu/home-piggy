"""Typed, validated model configuration (fail-closed).

All market parameters come from a YAML file (produced by the calibration step
or written by hand for tests). Nothing is defaulted silently: a missing or
out-of-range field raises ``ConfigError``. Units are stated next to every
field, and *annual* units are converted to *monthly* in exactly one place
(``market.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .regimes import validate_transition_matrix

#: Order of the jointly-Gaussian innovations; correlation matrices use it.
STATE_VARS: tuple[str, ...] = (
    "equity",         # monthly log total return of global equity (EUR)
    "house",          # monthly log growth of the NL house price index
    "short_rate",     # EUR short rate (cash), level
    "term_spread",    # 2y government yield minus short rate, level
    "mortgage_rate",  # NL 10y-fixed mortgage rate, level
    "inflation",      # NL/EA HICP inflation, annualised level
)
LEVEL_VARS: tuple[str, ...] = ("short_rate", "term_spread", "mortgage_rate", "inflation")


class ConfigError(ValueError):
    """Raised when a configuration is missing fields or is internally inconsistent."""


def _req(d: Mapping[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ConfigError(f"Missing required field '{ctx}.{key}'.")
    return d[key]


def _num(x: Any, ctx: str, lo: float | None = None, hi: float | None = None) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{ctx} must be a number, got {x!r}.") from exc
    if not np.isfinite(v):
        raise ConfigError(f"{ctx} must be finite.")
    if lo is not None and v < lo:
        raise ConfigError(f"{ctx}={v} is below the minimum {lo}.")
    if hi is not None and v > hi:
        raise ConfigError(f"{ctx}={v} is above the maximum {hi}.")
    return v


@dataclass(frozen=True)
class GrowthVar:
    """Per-regime parameters of a growth variable (equity, house)."""

    mu: np.ndarray      # (K,) mean *log* growth, per year
    sigma: np.ndarray   # (K,) volatility of log growth innovations, per sqrt(year)


@dataclass(frozen=True)
class LevelVar:
    """Per-regime parameters of a mean-reverting level (rates, inflation)."""

    theta: np.ndarray      # (K,) long-run level the variable reverts to, in this regime
    sigma: np.ndarray      # (K,) volatility of monthly changes, annualised (x sqrt(12))
    half_life_months: float
    initial: float         # level at the start date


@dataclass(frozen=True)
class ImpliedVolParams:
    vrp_ratio: float       # implied ATM vol / regime realised equity vol
    skew: float            # vol points per unit of -ln(K/S)
    min_vol: float         # floor on implied volatility


@dataclass(frozen=True)
class MarketParams:
    regimes: tuple[str, ...]
    P: np.ndarray                       # (K, K) monthly transition matrix
    initial_regime: str | np.ndarray    # regime name, or (K,) probabilities
    equity: GrowthVar
    house: GrowthVar
    house_ar1: float                    # phi: AR(1) persistence of monthly house growth
    levels: dict[str, LevelVar]
    corr: np.ndarray                    # (K, 6, 6) innovation correlations per regime
    implied_vol: ImpliedVolParams
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.regimes)

    def initial_probs(self) -> np.ndarray:
        if isinstance(self.initial_regime, str):
            p = np.zeros(self.k)
            p[self.regimes.index(self.initial_regime)] = 1.0
            return p
        return np.asarray(self.initial_regime, dtype=float)


def _per_regime(x: Any, k: int, ctx: str, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(x, dtype=float))
    if arr.shape != (k,):
        raise ConfigError(f"{ctx} needs one value per regime ({k}), got shape {arr.shape}.")
    for i, v in enumerate(arr):
        _num(v, f"{ctx}[{i}]", lo, hi)
    return arr


def _validate_corr(C: np.ndarray, ctx: str) -> np.ndarray:
    n = len(STATE_VARS)
    if C.shape != (n, n):
        raise ConfigError(f"{ctx} must be {n}x{n} in order {STATE_VARS}.")
    if not np.allclose(C, C.T, atol=1e-10):
        raise ConfigError(f"{ctx} must be symmetric.")
    if not np.allclose(np.diag(C), 1.0):
        raise ConfigError(f"{ctx} must have a unit diagonal.")
    if np.any(np.abs(C) > 1.0 + 1e-12):
        raise ConfigError(f"{ctx} has entries outside [-1, 1].")
    return C


def market_params_from_dict(d: Mapping[str, Any]) -> MarketParams:
    regimes = tuple(_req(d, "regimes", "market"))
    k = len(regimes)
    if k < 1 or len(set(regimes)) != k:
        raise ConfigError("market.regimes must be a non-empty list of unique names.")
    try:
        P = validate_transition_matrix(np.asarray(_req(d, "transition_matrix", "market"), dtype=float))
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if P.shape[0] != k:
        raise ConfigError("transition_matrix size does not match regimes.")

    init = _req(d, "initial_regime", "market")
    if isinstance(init, str):
        if init not in regimes:
            raise ConfigError(f"initial_regime '{init}' is not one of {regimes}.")
    else:
        init = _per_regime(init, k, "market.initial_regime", 0.0, 1.0)
        if not np.isclose(init.sum(), 1.0):
            raise ConfigError("initial_regime probabilities must sum to 1.")

    eq = _req(d, "equity", "market")
    equity = GrowthVar(
        _per_regime(_req(eq, "mu", "equity"), k, "equity.mu", -1.0, 1.0),
        _per_regime(_req(eq, "sigma", "equity"), k, "equity.sigma", 0.0, 2.0),
    )
    ho = _req(d, "house", "market")
    house = GrowthVar(
        _per_regime(_req(ho, "mu", "house"), k, "house.mu", -1.0, 1.0),
        _per_regime(_req(ho, "sigma", "house"), k, "house.sigma", 0.0, 1.0),
    )
    phi = _num(_req(ho, "ar1", "house"), "house.ar1", -0.99, 0.99)

    levels: dict[str, LevelVar] = {}
    for name in LEVEL_VARS:
        lv = _req(d, name, "market")
        levels[name] = LevelVar(
            theta=_per_regime(_req(lv, "theta", name), k, f"{name}.theta", -0.2, 0.5),
            sigma=_per_regime(_req(lv, "sigma", name), k, f"{name}.sigma", 0.0, 0.5),
            half_life_months=_num(_req(lv, "half_life_months", name), f"{name}.half_life_months", 0.5, 1200.0),
            initial=_num(_req(lv, "initial", name), f"{name}.initial", -0.2, 0.5),
        )

    corr_raw = _req(d, "correlation", "market")
    if isinstance(corr_raw, Mapping):
        mats = []
        for r in regimes:
            if r not in corr_raw:
                raise ConfigError(f"correlation missing for regime '{r}'.")
            mats.append(_validate_corr(np.asarray(corr_raw[r], dtype=float), f"correlation[{r}]"))
        corr = np.stack(mats)
    else:
        C = _validate_corr(np.asarray(corr_raw, dtype=float), "correlation")
        corr = np.repeat(C[None], k, axis=0)

    iv = _req(d, "implied_vol", "market")
    implied = ImpliedVolParams(
        vrp_ratio=_num(_req(iv, "vrp_ratio", "implied_vol"), "implied_vol.vrp_ratio", 0.5, 3.0),
        skew=_num(_req(iv, "skew", "implied_vol"), "implied_vol.skew", 0.0, 2.0),
        min_vol=_num(_req(iv, "min_vol", "implied_vol"), "implied_vol.min_vol", 0.001, 1.0),
    )
    return MarketParams(
        regimes=regimes, P=P, initial_regime=init, equity=equity, house=house, house_ar1=phi,
        levels=levels, corr=corr, implied_vol=implied, provenance=dict(d.get("provenance", {})),
    )


def load_market_params(path: str | Path) -> MarketParams:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, Mapping) or "market" not in raw:
        raise ConfigError(f"{path}: expected a top-level 'market' mapping.")
    params = market_params_from_dict(raw["market"])
    prov = dict(raw.get("provenance", {}))
    if not prov.get("sources") and not prov.get("synthetic", False):
        raise ConfigError(
            f"{path}: provenance.sources is empty. Every parameter file must cite its data "
            "sources, or declare provenance.synthetic: true (tests only)."
        )
    return MarketParams(**{**params.__dict__, "provenance": prov})


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
