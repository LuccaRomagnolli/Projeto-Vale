import pandas as pd
from src.features.build_features import (
    add_alert_history_features,
    build_feature_dataset,
)


def test_alert_history_features_use_only_past_events() -> None:
    df = pd.DataFrame(
        {
            "Tag": ["A", "A"],
            "Fim": ["2026-01-01 10:00:00+00:00", "2026-01-01 12:00:00+00:00"],
            "Inicio": ["2026-01-01 09:30:00+00:00", "2026-01-01 11:30:00+00:00"],
            "Classe": ["Operando", "Operando"],
            "duracao_ciclo_min": [30.0, 30.0],
            "target_4h": [1, 0],
        }
    )
    critical = pd.DataFrame(
        {
            "TAG": ["A", "A"],
            "EVENT_TIME": ["2026-01-01 11:00:00+00:00", "2026-01-01 13:00:00+00:00"],
        }
    )

    out = add_alert_history_features(df, critical)

    assert int(out.loc[0, "n_alertas_4h"]) == 0
    assert int(out.loc[1, "n_alertas_4h"]) == 1
    assert round(float(out.loc[1, "dias_desde_ultimo_alerta"]), 4) == round(1 / 24, 4)


def test_alert_window_boundary_is_measured_in_hours() -> None:
    """A janela de N horas precisa excluir eventos anteriores a ela.

    Regressao: o codigo derivava o timestamp inteiro de `astype("int64")` e
    subtraia uma constante de nanossegundos. Quando o pandas 3.0 mudou a
    resolucao padrao para microssegundos, a janela de 4h passou a valer ~166
    dias sem qualquer erro. O teste antigo nao pegava isso porque so havia
    eventos dentro da janela em ambos os casos.
    """
    reference = pd.Timestamp("2026-01-02 12:00:00+00:00")
    df = pd.DataFrame(
        {
            "Tag": ["A"],
            "Fim": [reference],
            "Inicio": [reference - pd.Timedelta(minutes=30)],
            "Classe": ["Operando"],
            "duracao_ciclo_min": [30.0],
            "target_4h": [0],
        }
    )
    critical = pd.DataFrame(
        {
            "TAG": ["A", "A", "A"],
            "EVENT_TIME": [
                reference - pd.Timedelta(hours=3),  # dentro de 4h e de 24h
                reference - pd.Timedelta(hours=10),  # so dentro de 24h
                reference - pd.Timedelta(days=30),  # fora de todas as janelas
            ],
        }
    )

    out = add_alert_history_features(df, critical)

    assert int(out.loc[0, "n_alertas_4h"]) == 1
    assert int(out.loc[0, "n_alertas_24h"]) == 2
    assert round(float(out.loc[0, "dias_desde_ultimo_alerta"]), 4) == round(3 / 24, 4)


def test_build_feature_dataset_adds_model_ready_columns() -> None:
    labeled = pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "Inicio": [
                "2026-01-01 08:00:00+00:00",
                "2026-01-01 09:00:00+00:00",
                "2026-01-01 12:00:00+00:00",
            ],
            "Fim": [
                "2026-01-01 08:30:00+00:00",
                "2026-01-01 09:30:00+00:00",
                "2026-01-01 12:30:00+00:00",
            ],
            "Tag": ["A", "A", "B"],
            "Frota": ["F1", "F1", "F2"],
            "Tipo": ["Caminhao", "Caminhao", "Carregadeira"],
            "Classe": ["Operando", "Parado", "Operando"],
            "target_4h": [1, 0, 1],
        }
    )
    critical = pd.DataFrame(
        {
            "TAG": ["A", "B"],
            "EVENT_TIME": ["2026-01-01 07:30:00+00:00", "2026-01-01 11:30:00+00:00"],
        }
    )

    features = build_feature_dataset(labeled, critical)

    expected_columns = {
        "hora_do_dia",
        "turno",
        "duracao_ciclo_min",
        "n_ciclos_4h",
        "duracao_std_ciclo_4h",
        "n_classes_distintas_4h",
        "n_alertas_4h",
        "alertas_por_hora_4h",
        "dias_desde_ultimo_alerta",
        "delta_duracao_ciclo_4h_24h",
        "ratio_alertas_4h_24h",
    }
    assert expected_columns.issubset(set(features.columns))
    assert int(features["target_4h"].sum()) == 2

    # Os encodings categoricos saem do build e passam a ser ajustados no treino
    # depois do split (src/features/encoders.py). As colunas cruas precisam
    # sobreviver para que o encoder possa aprender com elas.
    encoded_columns = {"Tag_freq", "Operador_freq", "Classe_target_enc"}
    assert not encoded_columns & set(features.columns)
    assert {"Tag", "Classe", "Frota", "Tipo"}.issubset(set(features.columns))
