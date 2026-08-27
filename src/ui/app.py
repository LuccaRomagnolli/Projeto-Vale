"""Aplicacao FastAPI da interface operacional de engenharia de minas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.ui.service import OperationalStore

STATIC_DIR = Path(__file__).resolve().parent / "static"


def asset_version() -> str:
    """Carimbo derivado do mtime dos assets, usado para invalidar cache do navegador."""
    stamps = [
        path.stat().st_mtime
        for path in (STATIC_DIR / "app.css", STATIC_DIR / "app.js")
        if path.exists()
    ]
    return str(int(max(stamps))) if stamps else "0"


class ActionRequest(BaseModel):
    item_type: str = Field(pattern="^(priority|event|cycle)$")
    status: str
    tag: str | None = None
    date: str | None = None
    event_time: str | None = None
    cycle_id: str | None = None
    note: str = ""
    operator: str = ""


def create_app(store: OperationalStore | None = None) -> FastAPI:
    store = store or OperationalStore()
    app = FastAPI(
        title="Vale · Engenharia de Minas",
        description="Console operacional de alertas criticos Don't Go e processamento da frota.",
        version="1.0.0",
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/filters")
    def filters() -> dict[str, Any]:
        return store.filters()

    @app.get("/api/overview")
    def overview(date: str | None = Query(default=None)) -> dict[str, Any]:
        return store.overview(selected_date=date)

    @app.get("/api/priority")
    def priority(
        date: str | None = Query(default=None),
        frota: str | None = Query(default=None),
        risco: str | None = Query(default=None),
        q: str | None = Query(default=None),
    ) -> dict[str, Any]:
        return store.priority_board(selected_date=date, frota=frota, risco=risco, query=q)

    @app.get("/api/alerts")
    def alerts(
        date: str | None = Query(default=None),
        tag: str | None = Query(default=None),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        return store.alerts(selected_date=date, tag=tag, page=page, page_size=page_size)

    @app.get("/api/cycles")
    def cycles(
        date: str | None = Query(default=None),
        tag: str | None = Query(default=None),
        frota: str | None = Query(default=None),
        classe: str | None = Query(default=None),
        target: str | None = Query(default=None),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        return store.cycles(
            selected_date=date,
            tag=tag,
            frota=frota,
            classe=classe,
            target=target,
            page=page,
            page_size=page_size,
        )

    @app.get("/api/processing")
    def processing(date: str | None = Query(default=None)) -> dict[str, Any]:
        return store.processing_summary(selected_date=date)

    @app.get("/api/equipment/{tag}")
    def equipment(tag: str) -> dict[str, Any]:
        return store.equipment(tag)

    @app.get("/api/performance")
    def performance() -> dict[str, Any]:
        return store.performance()

    @app.get("/api/actions")
    def list_actions() -> dict[str, Any]:
        return {"items": store.actions()}

    @app.post("/api/actions")
    def save_action(payload: ActionRequest) -> dict[str, Any]:
        try:
            return store.upsert_action(
                item_type=payload.item_type,
                status=payload.status,
                tag=payload.tag,
                selected_date=payload.date,
                event_time=payload.event_time,
                cycle_id=payload.cycle_id,
                note=payload.note,
                operator=payload.operator,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/reload")
    def reload_store() -> dict[str, str]:
        store.reload()
        return {"status": "reloaded"}

    @app.get("/brand/logo.png")
    def logo() -> FileResponse:
        if not store.paths.logo.exists():
            raise HTTPException(status_code=404, detail="Logo nao encontrada")
        return FileResponse(store.paths.logo)

    @app.get("/")
    def index() -> HTMLResponse:
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        html = html.replace("__ASSET_VERSION__", asset_version())
        return HTMLResponse(html, headers={"Cache-Control": "no-store, must-revalidate"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.state.store = store
    return app


app = create_app()
