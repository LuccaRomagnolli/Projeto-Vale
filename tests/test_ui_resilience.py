"""Resiliencia do backend: artefato ruim degrada, nao derruba."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from src.ui.app import create_app
from src.ui.service import OperationalStore, StorePaths


def _paths(tmp_path: Path) -> StorePaths:
    return StorePaths(
        priority=tmp_path / "priority.csv",
        labeled=tmp_path / "labeled.parquet",
        events=tmp_path / "events.parquet",
        metrics=tmp_path / "metrics.json",
        hotspots=tmp_path / "hotspots.csv",
        selection=tmp_path / "selection.json",
        worklog=tmp_path / "worklog.json",
        logo=tmp_path / "logo.png",
    )


def _seed(tmp_path: Path) -> StorePaths:
    paths = _paths(tmp_path)
    pd.DataFrame(
        {
            "data": ["2026-05-01"] * 2,
            "rank": [1, 2],
            "Tag": ["CA1", "CA2"],
            "score": [0.9, 0.8],
            "risco_segmento": ["alto_acima_threshold", "alto_por_ranking"],
        }
    ).to_csv(paths.priority, index=False)
    pd.DataFrame(
        {
            "Id": [1, 2],
            "Inicio": pd.to_datetime(["2026-05-01 08:00", "2026-05-01 09:00"], utc=True),
            "Fim": pd.to_datetime(["2026-05-01 08:30", "2026-05-01 09:30"], utc=True),
            "Tag": ["CA1", "CA2"],
            "Frota": ["F1", "F1"],
            "Tipo": ["Caminhao", "Caminhao"],
            "Classe": ["Operando", "Parado"],
            "tte_horas": [1.0, 2.0],
            "target_4h": [1, 0],
        }
    ).to_parquet(paths.labeled, index=False)
    pd.DataFrame(
        {"TAG": ["CA1"], "EVENT_TIME": pd.to_datetime(["2026-05-01 10:00"], utc=True)}
    ).to_parquet(paths.events, index=False)
    paths.metrics.write_text(json.dumps({"model_name": "m", "threshold": 0.3}), encoding="utf-8")
    paths.selection.write_text(json.dumps({"selected_model": {}}), encoding="utf-8")
    return paths


# --- degradacao em vez de queda ------------------------------------------


def test_corrupt_parquet_degrades_instead_of_crashing(tmp_path: Path) -> None:
    """Antes, um parquet truncado levantava excecao no import e o servidor nao subia."""
    paths = _seed(tmp_path)
    paths.labeled.write_bytes(b"isto nao e um parquet")

    store = OperationalStore(paths)  # nao deve levantar

    health = store.health()
    assert health["status"] == "degraded"
    assert "cycles" in health["degraded_sources"]
    assert health["sources"]["cycles"]["error"]
    # o painel de prioridade, que nao depende do parquet, continua servindo
    assert store.priority_board(selected_date="2026-05-01")["count"] == 2


def test_corrupt_json_degrades_only_its_own_panel(tmp_path: Path) -> None:
    paths = _seed(tmp_path)
    paths.metrics.write_text("{ json quebrado", encoding="utf-8")

    store = OperationalStore(paths)

    assert "metrics" in store.health()["degraded_sources"]
    assert store.overview(selected_date="2026-05-01")["kpis"]["priority_count"] == 2


def test_health_endpoint_reports_degradation(tmp_path: Path) -> None:
    paths = _seed(tmp_path)
    paths.hotspots.write_text("a,b\n1", encoding="utf-8")
    paths.selection.unlink()

    client = TestClient(create_app(OperationalStore(paths)))
    payload = client.get("/api/health").json()

    assert payload["status"] == "degraded"
    assert "selection" in payload["degraded_sources"]


def test_health_is_ok_when_every_artifact_loads(tmp_path: Path) -> None:
    paths = _seed(tmp_path)
    paths.hotspots.write_text("Tag,positive_days\nCA1,3\n", encoding="utf-8")
    client = TestClient(create_app(OperationalStore(paths)))
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["degraded_sources"] == []


# --- worklog ------------------------------------------------------------


def test_corrupt_worklog_does_not_break_endpoints(tmp_path: Path) -> None:
    """Worklog malformado fazia todo endpoint que o consulta devolver 500."""
    paths = _seed(tmp_path)
    paths.worklog.write_text("{ nao e json", encoding="utf-8")
    store = OperationalStore(paths)

    assert store.actions() == []
    assert store.overview(selected_date="2026-05-01")["kpis"]["open_actions"] == 0
    assert "worklog" in store.load_errors()


def test_worklog_stored_as_list_is_tolerated(tmp_path: Path) -> None:
    """O guard `isinstance` vinha depois do `.get` e nunca protegia nada."""
    paths = _seed(tmp_path)
    paths.worklog.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    store = OperationalStore(paths)
    assert store.actions() == []


def test_worklog_write_is_atomic(tmp_path: Path) -> None:
    """Nenhum estado intermediario invalido deve ficar visivel no arquivo."""
    paths = _seed(tmp_path)
    store = OperationalStore(paths)
    store.upsert_action(
        item_type="priority", status="pendente", tag="CA1", selected_date="2026-05-01"
    )

    observed: list[bool] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            if paths.worklog.exists():
                try:
                    json.loads(paths.worklog.read_text(encoding="utf-8"))
                    observed.append(True)
                except json.JSONDecodeError:
                    observed.append(False)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        for index in range(40):
            store.upsert_action(
                item_type="priority",
                status="em_inspecao",
                tag=f"CA{index}",
                selected_date="2026-05-01",
                note="x" * 400,
            )
    finally:
        stop.set()
        thread.join(timeout=5)

    assert observed, "o leitor concorrente nao chegou a observar o arquivo"
    assert all(observed), "uma leitura concorrente encontrou JSON parcial"


def test_worklog_leaves_no_temporary_files(tmp_path: Path) -> None:
    paths = _seed(tmp_path)
    store = OperationalStore(paths)
    store.upsert_action(
        item_type="priority", status="pendente", tag="CA1", selected_date="2026-05-01"
    )
    leftovers = [p.name for p in paths.worklog.parent.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


# --- snapshot coerente ---------------------------------------------------


def test_reload_swaps_generations_atomically(tmp_path: Path) -> None:
    paths = _seed(tmp_path)
    store = OperationalStore(paths)
    first = store._snapshot

    store.reload()

    assert store._snapshot is not first, "reload deve produzir uma nova geracao"
    # a geracao anterior permanece integra para quem ja a tinha em maos
    assert len(first.priority) == 2
