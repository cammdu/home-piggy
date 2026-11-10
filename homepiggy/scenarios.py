"""Narrative scenarios = forced regime segments + deterministic shocks.

A scenario is pure data. The core ships only the neutral ``baseline`` (no
forcing, no shocks). Historical-replay scenarios (2008-style, 2022 rate shock,
NL housing correction, ...) are produced by the calibration step from observed
data series, with the source of every magnitude recorded in the scenario file.
No shock size is ever typed in by hand without a source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .market import ShockEvent
from .regimes import RegimeSegment


@dataclass(frozen=True)
class Scenario:
    name: str = "baseline"
    description: str = "Free-running Markov regimes, no shocks."
    segments: tuple[RegimeSegment, ...] = ()
    shocks: tuple[ShockEvent, ...] = ()
    sources: tuple[str, ...] = ()
    #: If set, scenario months are counted back from the purchase date
    #: (e.g. a shock "in the final 2 years"), which is how sequence risk is shown.
    anchor: str = "start"          # 'start' or 'purchase'

    def resolve(self, horizon_months: int) -> "Scenario":
        """Convert purchase-anchored months to start-anchored months."""
        if self.anchor == "start":
            return self
        if self.anchor != "purchase":
            raise ValueError("anchor must be 'start' or 'purchase'.")
        segs = tuple(
            RegimeSegment(horizon_months - s.end, horizon_months - s.start, s.regime) for s in self.segments
        )
        shocks = tuple(
            ShockEvent(**{**e.__dict__, "month": horizon_months - e.month - e.duration_months})
            for e in self.shocks
        )
        return Scenario(self.name, self.description, segs, shocks, self.sources, "start")


def scenario_from_dict(d: Mapping[str, Any]) -> Scenario:
    if d.get("shocks") and not d.get("sources"):
        raise ValueError(f"Scenario '{d.get('name')}' has shocks but no sources; every magnitude must be sourced.")
    return Scenario(
        name=str(d["name"]),
        description=str(d.get("description", "")),
        segments=tuple(RegimeSegment(int(s["start"]), int(s["end"]), str(s["regime"])) for s in d.get("segments", [])),
        shocks=tuple(ShockEvent(**s) for s in d.get("shocks", [])),
        sources=tuple(d.get("sources", [])),
        anchor=str(d.get("anchor", "start")),
    )


BASELINE = Scenario()
