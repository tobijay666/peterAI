"""Usage -> visual style rules for the 3-D mind view (spec Phase 5A).

The server decides how every node looks; the browser only draws. Keeping the
rules here (pure functions, no I/O) means they are unit- and property-tested
in Python:

  * high usage   -> larger, brightly emissive (bloom), fully saturated
  * moderate     -> baseline size and colour
  * low usage    -> smaller, dull, desaturated, semi-transparent
  * fading       -> shrinking toward the prune size, nearly invisible

Values interpolate inside each tier so size changes smoothly.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from core.cognition_config import cognition_config

TIERS = ("high", "moderate", "low", "fading")


def _lerp(a: float, b: float, t: float) -> float:
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t


def tier_for(percentile: float, status: str = "active", cfg: Optional[Dict[str, Any]] = None) -> str:
    v = (cfg or cognition_config())["visual"]
    if status in ("fading", "pruned"):
        return "fading"
    if percentile >= v["tiers"]["high_percentile"]:
        return "high"
    if percentile < v["tiers"]["low_percentile"]:
        return "low"
    return "moderate"


def visual_style(percentile: float, status: str = "active",
                 cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Map a usage percentile in [0, 1] (and node status) to render parameters.

    Returns {tier, scale, emissive, saturation, opacity}. Monotonic: for an
    active node, scale/emissive/saturation/opacity never decrease as the
    percentile rises.
    """
    c = cfg or cognition_config()
    v = c["visual"]
    p = max(0.0, min(1.0, float(percentile or 0.0)))
    tier = tier_for(p, status, c)
    hi_cut = v["tiers"]["high_percentile"]
    lo_cut = v["tiers"]["low_percentile"]

    if tier == "fading":
        f = v["fading"]
        return {"tier": "fading", "scale": f["scale"], "emissive": f["emissive"],
                "saturation": f["saturation"], "opacity": f["opacity"]}
    if tier == "high":
        h = v["high"]
        t = (p - hi_cut) / max(1e-9, 1.0 - hi_cut)
        return {"tier": "high",
                "scale": round(_lerp(h["scale_min"], h["scale_max"], t), 4),
                "emissive": round(_lerp(h["emissive_min"], h["emissive_max"], t), 4),
                "saturation": h["saturation"], "opacity": h["opacity"]}
    if tier == "low":
        lo = v["low"]
        t = p / max(1e-9, lo_cut)
        return {"tier": "low",
                "scale": round(_lerp(lo["scale_min"], lo["scale_max"], t), 4),
                "emissive": lo["emissive"], "saturation": lo["saturation"],
                "opacity": round(_lerp(lo["opacity_min"], lo["opacity_max"], t), 4)}
    m = v["moderate"]
    return {"tier": "moderate", "scale": m["scale"], "emissive": m["emissive"],
            "saturation": m["saturation"], "opacity": m["opacity"]}
