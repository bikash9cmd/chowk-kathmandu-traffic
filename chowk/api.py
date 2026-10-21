"""HTTP API (FastAPI, local). The same routes run on Lambda in chowk/aws/handlers.py."""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException

from chowk.status import approach_status, junctions
from chowk.store import FlowStore, LocalStore


def create_app(store: FlowStore | None = None) -> FastAPI:
    store = store or LocalStore()
    app = FastAPI(title="Chowk", description="15-minute PCU flows at Kathmandu junctions")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/junctions")
    def list_junctions() -> dict:
        return {k: v["name"] for k, v in junctions().items()} | {"replay": "Replayed recording sessions"}

    @app.get("/junctions/{junction_id}/now")
    def now(junction_id: str) -> dict:
        j = junctions().get(junction_id)
        if j is None and junction_id != "replay":
            raise HTTPException(404, f"unknown junction {junction_id}")
        approaches = j["approaches"] if j else {"recorded_clips": None}
        return {"junction": junction_id,
                "approaches": [approach_status(store, junction_id, a, g) for a, g in approaches.items()]}

    @app.get("/junctions/{junction_id}/{approach_id}/flows")
    def flows(junction_id: str, approach_id: str, since: str | None = None, hours: int = 24) -> dict:
        if not 1 <= hours <= 24 * 31:
            raise HTTPException(400, "hours must be between 1 and 744")
        start = since or "0000"
        until = (datetime.fromisoformat(since) + timedelta(hours=hours)).isoformat() if since else "9999"
        return {"junction": junction_id, "approach": approach_id,
                "flows": store.query(junction_id, approach_id, start, until)}

    return app
