"""What the API says about an approach right now. No web framework, so Lambda can import it."""

from __future__ import annotations

import json

from chowk import config
from chowk.counting import congestion
from chowk.store import FlowStore


def junctions() -> dict:
    return json.loads((config.DATA_DIR / "junctions.json").read_text())["junctions"]


def approach_status(store: FlowStore, junction_id: str, approach_id: str, geometry: dict | None) -> dict:
    rec = store.latest(junction_id, approach_id)
    if rec is None:
        return {"approach": approach_id, "status": "no_data"}
    out = {"approach": approach_id, "interval_start": rec["interval_start"], "pcu_per_hour": rec["pcu_per_hour"],
           "counts": rec["counts"], "minutes_observed": rec["minutes_observed"], "model_version": rec["model_version"]}
    # Under half an interval observed: the rate is an extrapolation from a few minutes. Say so.
    if rec["minutes_observed"] < config.FLOW_INTERVAL_MINUTES / 2:
        out["status"] = "partial"
    # Most frames dark, blurry or compressed: the classifier is outside what it was tested on
    # (docs/FAILURE_MODES.md). Report the numbers, but don't let anyone mistake them for clean ones.
    bad = sorted(f for f, share in rec.get("image_quality", {}).items() if share > 0.5)
    if bad:
        out["status"] = "degraded_image"
        out["image_problems"] = bad
    if geometry:
        vc, band = congestion(rec["pcu_per_hour"], geometry["lanes"], geometry["green_ratio"],
                              geometry.get("saturation_pcu_per_lane_hour", config.DEFAULT_SATURATION_PCU_PER_LANE_HOUR))
        out |= {"volume_to_capacity": vc, "level": band}
    out.setdefault("status", "ok")
    return out
