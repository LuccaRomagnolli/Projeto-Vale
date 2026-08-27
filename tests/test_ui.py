from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from src.ui.app import create_app
from src.ui.service import OperationalStore, StorePaths, paginate


def _write_store(tmp_path: Path) -> OperationalStore:
    labeled = pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "Inicio": pd.to_datetime(
                ["2025-06-02T10:00:00Z", "2025-06-02T12:00:00Z", "2025-06-03T08:00:00Z"]
            ),
            "Fim": pd.to_datetime(
                ["2025-06-02T11:00:00Z", "2025-06-02T13:30:00Z", "2025-06-03T09:00:00Z"]
            ),
            "Tag": ["CA1", "CA1", "CA2"],
            "Frota": ["793-D 5S", "793-D 5S", "793-D 2S"],
            "Tipo": ["Caminhao", "Caminhao", "Caminhao"],
            "Classe": ["Carga", "Manobra", "Carga"],
            "tte_horas": [1.5, 3.0, 8.0],
            "target_4h": [1, 0, 0],
        }
    )
    events = pd.DataFrame(
        {
            "TAG": ["CA1", "CA2"],
            "EVENT_TIME": pd.to_datetime(["2025-06-02T12:10:00Z", "2025-06-03T10:00:00Z"]),
        }
    )
    priority = pd.DataFrame(
        {
            "data": ["2025-06-02", "2025-06-02"],
            "rank": [1, 2],
            "Tag": ["CA1", "CA2"],
            "score": [0.81, 0.44],
            "Frota": ["793-D 5S", "793-D 2S"],
            "Tipo": ["Caminhao", "Caminhao"],
            "turno": ["tarde", "noite"],
            "motivo_principal": ["alertas recentes na janela de 4h", "maior score diario do modelo"],
            "risco_segmento": ["alto_acima_threshold", "monitorar_por_ranking"],
            "acao_recomendada": [
                "inspecionar primeiro e registrar achado operacional",
                "monitorar no painel e reavaliar no proximo ciclo",
            ],
        }
    )
    labeled_path = tmp_path / "labeled.parquet"
    events_path = tmp_path / "events.parquet"
    priority_path = tmp_path / "priority.csv"
    labeled.to_parquet(labeled_path, index=False)
    events.to_parquet(events_path, index=False)
    priority.to_csv(priority_path, index=False)
    paths = StorePaths(
        priority=priority_path,
        labeled=labeled_path,
        events=events_path,
        metrics=tmp_path / "missing_metrics.json",
        hotspots=tmp_path / "missing_hotspots.csv",
        selection=tmp_path / "missing_selection.json",
        worklog=tmp_path / "worklog.json",
        logo=tmp_path / "missing-logo.png",
    )
    return OperationalStore(paths=paths)


def test_paginate_respects_page_bounds() -> None:
    frame = pd.DataFrame({"n": list(range(5))})
    page, total = paginate(frame, page=2, page_size=2)
    assert total == 5
    assert page["n"].tolist() == [2, 3]


def test_overview_and_priority_board(tmp_path: Path) -> None:
    store = _write_store(tmp_path)
    overview = store.overview("2025-06-02")
    assert overview["kpis"]["priority_count"] == 2
    assert overview["kpis"]["high_risk_count"] == 1
    assert overview["kpis"]["cycle_count"] == 2
    assert overview["kpis"]["critical_events"] == 1

    board = store.priority_board(selected_date="2025-06-02", frota="793-D 5S")
    assert board["count"] == 1
    assert board["items"][0]["Tag"] == "CA1"
    assert board["items"][0]["status"] == "pendente"


def test_action_roundtrip_and_api(tmp_path: Path) -> None:
    store = _write_store(tmp_path)
    action = store.upsert_action(
        item_type="priority",
        status="em_inspecao",
        tag="CA1",
        selected_date="2025-06-02",
        note="vazamento no conjunto",
        operator="turno A",
    )
    assert action["status"] == "em_inspecao"
    board = store.priority_board(selected_date="2025-06-02")
    treated = next(item for item in board["items"] if item["Tag"] == "CA1")
    assert treated["status"] == "em_inspecao"

    client = TestClient(create_app(store))
    home = client.get("/")
    assert home.status_code == 200
    assert "Engenharia de minas" in home.text

    health = client.get("/api/health")
    assert health.json()["status"] == "ok"

    alerts = client.get("/api/alerts", params={"date": "2025-06-02"})
    assert alerts.status_code == 200
    assert alerts.json()["total"] == 1

    cycles = client.get("/api/cycles", params={"date": "2025-06-02", "tag": "CA1"})
    assert cycles.json()["total"] == 2

    saved = client.post(
        "/api/actions",
        json={
            "item_type": "event",
            "status": "resolvido",
            "tag": "CA1",
            "event_time": "2025-06-02T12:10:00+00:00",
            "operator": "manutencao",
        },
    )
    assert saved.status_code == 200
    assert saved.json()["status"] == "resolvido"

    invalid = client.post(
        "/api/actions",
        json={"item_type": "priority", "status": "desconhecido", "tag": "CA1", "date": "2025-06-02"},
    )
    assert invalid.status_code == 400
