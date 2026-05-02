import pandas as pd
from src.alert_labeler import (
    build_critical_events,
    build_critical_events_from_source,
    build_targets_from_events,
)


def test_build_critical_events_uses_rule_and_fallback() -> None:
    events_df = pd.DataFrame(
        {
            "TAG": ["A", "B", "C"],
            "Data_Evento": [
                "2026-01-01 10:00:00",
                "2026-01-01 11:00:00",
                "2026-01-01 12:00:00",
            ],
            "Alarme": ["Overheat", "Unknown Alarm", "Another Alarm"],
            "Is_Dont_Go": [0, 1, 0],
        }
    )
    rules_df = pd.DataFrame(
        {
            "TIPO": ["ALARME OEM"],
            "EVENTO": ["Overheat"],
            "SITUACAO": ["Mediante alarme nível 3"],
            "NIVEL": ["Muito Alto"],
        }
    )

    critical, audit = build_critical_events(events_df=events_df, rules_df=rules_df)

    assert len(critical) == 2
    assert audit["rows_match_event_only"] == 1
    assert audit["rows_match_is_dont_go"] == 1


def test_build_targets_from_events_sets_target_4h_and_tte() -> None:
    apont = pd.DataFrame(
        {
            "Tag": ["A", "A", "A", "B"],
            "Fim": [
                "2026-01-01 08:00:00",
                "2026-01-01 09:30:00",
                "2026-01-01 15:00:00",
                "2026-01-01 10:00:00",
            ],
        }
    )
    critical = pd.DataFrame(
        {
            "TAG": ["A", "A", "B"],
            "EVENT_TIME": [
                "2026-01-01 10:00:00",
                "2026-01-01 10:30:00",
                "2026-01-01 18:30:00",
            ],
        }
    )

    out = build_targets_from_events(
        apontamentos_df=apont, critical_events_df=critical, horizon_hours=4
    )
    out = out.sort_values(["Tag", "Fim"]).reset_index(drop=True)

    assert out.loc[0, "target_4h"] == 1
    assert round(out.loc[0, "tte_horas"], 2) == 2.0
    assert out.loc[1, "target_4h"] == 1
    assert round(out.loc[1, "tte_horas"], 2) == 0.5
    assert out.loc[2, "target_4h"] == 0
    assert pd.isna(out.loc[2, "tte_horas"])
    assert out.loc[3, "target_4h"] == 0


def test_build_targets_from_events_multiple_alerts_keeps_binary_target() -> None:
    apont = pd.DataFrame({"Tag": ["A"], "Fim": ["2026-01-01 08:00:00"]})
    critical = pd.DataFrame(
        {
            "TAG": ["A", "A", "A"],
            "EVENT_TIME": [
                "2026-01-01 09:00:00",
                "2026-01-01 10:00:00",
                "2026-01-01 11:00:00",
            ],
        }
    )

    out = build_targets_from_events(
        apontamentos_df=apont, critical_events_df=critical, horizon_hours=4
    )
    assert out.loc[0, "target_4h"] == 1
    assert round(out.loc[0, "tte_horas"], 2) == 1.0


def test_build_critical_events_from_source_directory(tmp_path) -> None:
    rules_df = pd.DataFrame(
        {
            "TIPO": ["ALARME OEM"],
            "EVENTO": ["Overheat"],
            "SITUACAO": ["Mediante alarme nível 3"],
            "NIVEL": ["Muito Alto"],
        }
    )
    telemetry_dir = tmp_path / "telemetria"
    telemetry_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "TAG": ["X1", "X2"],
            "Data_Evento": ["2025-01-01 00:10:00", "2025-01-01 00:20:00"],
            "Alarme": ["Overheat", "Other"],
            "Tipo": ["Caminhao", "Caminhao"],
            "Criticidade": ["Informacional", "Informacional"],
            "Is_Dont_Go": [0, 1],
        }
    )
    df.to_parquet(telemetry_dir / "telemetry_jan.parquet", index=False)

    critical, audit = build_critical_events_from_source(
        rules_df=rules_df,
        events_source=telemetry_dir,
    )

    assert len(critical) == 2
    assert audit["files_processed"] == 1
    assert audit["rows_total_events"] == 2
    assert audit["rows_match_event_only"] == 1
    assert audit["rows_match_is_dont_go"] == 1
