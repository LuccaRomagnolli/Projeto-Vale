"""Garantias contra vazamento temporal no split e nos encodings categoricos."""

from __future__ import annotations

import pandas as pd
import pytest
from src.features.encoders import CategoricalEncoder
from src.models.model_selection import make_backtest_folds
from src.models.validation import LABEL_HORIZON_HOURS, temporal_train_val_test_split


def _frame(n_rows: int = 400, start: str = "2026-01-01") -> pd.DataFrame:
    times = pd.date_range(start, periods=n_rows, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "Fim": times,
            "Tag": [f"T{i % 5}" for i in range(n_rows)],
            "Classe": ["Operando" if i % 3 else "Parado" for i in range(n_rows)],
            "Frota": [f"F{i % 2}" for i in range(n_rows)],
            "Tipo": ["Caminhao" if i % 2 else "Carregadeira" for i in range(n_rows)],
            "target_4h": [i % 4 == 0 for i in range(n_rows)],
        }
    )


def test_embargo_separates_splits_by_label_horizon() -> None:
    """Deve existir um intervalo >= horizonte do rotulo entre os blocos."""
    train, val, test, metadata = temporal_train_val_test_split(_frame())

    gap_train_val = val["Fim"].min() - train["Fim"].max()
    gap_val_test = test["Fim"].min() - val["Fim"].max()

    assert gap_train_val >= pd.Timedelta(hours=LABEL_HORIZON_HOURS)
    assert gap_val_test >= pd.Timedelta(hours=LABEL_HORIZON_HOURS)
    assert metadata["embargo_hours"] == LABEL_HORIZON_HOURS
    assert metadata["rows_dropped_embargo_train"] > 0
    assert metadata["rows_dropped_embargo_val"] > 0


def test_splits_are_chronologically_ordered_and_disjoint() -> None:
    train, val, test, _ = temporal_train_val_test_split(_frame())
    assert train["Fim"].max() < val["Fim"].min()
    assert val["Fim"].max() < test["Fim"].min()


def test_identical_timestamps_never_straddle_the_boundary() -> None:
    """Corte por calendario: ciclos simultaneos ficam do mesmo lado."""
    # 300 linhas distribuidas em 30 instantes, 10 ciclos simultaneos cada:
    # granularidade suficiente para tres blocos, com empates nas fronteiras.
    distinct = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
    frame = _frame(300)
    frame["Fim"] = distinct.repeat(10)

    train, val, test, _ = temporal_train_val_test_split(frame, embargo_hours=0)

    for part in (train, val, test):
        if not part.empty:
            # cada bloco contem instantes que nao aparecem em nenhum outro
            others = pd.concat([p for p in (train, val, test) if p is not part])
            assert not set(part["Fim"]) & set(others["Fim"])


def test_encoder_statistics_come_only_from_training_rows() -> None:
    """Uma Tag exclusiva de teste nao pode ganhar frequencia propria."""
    train = _frame(200)
    unseen = _frame(20, start="2026-06-01")
    unseen["Tag"] = "TAG_NOVA"
    unseen["Classe"] = "CLASSE_NOVA"

    encoder = CategoricalEncoder()
    encoder.fit_transform_train(train)
    encoded = encoder.transform(unseen)

    # Categoria nunca vista recebe valor neutro, nao estatistica do proprio lote.
    assert (encoded["Tag_freq"] == 0.0).all()
    assert (encoded["Classe_target_enc"] == encoder.global_target_mean).all()
    assert "TAG_NOVA" not in encoder.tag_freq


def test_encoder_is_invariant_to_the_rows_being_transformed() -> None:
    """Transformar em lote ou linha a linha deve dar o mesmo resultado.

    Se o encoding dependesse da composicao do proprio conjunto (como acontecia
    quando `value_counts` rodava sobre o dataset inteiro), estes valores
    divergiriam.
    """
    train = _frame(200)
    later = _frame(60, start="2026-05-01")

    encoder = CategoricalEncoder().fit(train)
    full = encoder.transform(later)
    piecewise = pd.concat([encoder.transform(later.iloc[[i]]) for i in range(len(later))])

    pd.testing.assert_series_equal(
        full["Tag_freq"].reset_index(drop=True),
        piecewise["Tag_freq"].reset_index(drop=True),
    )


def test_target_encoding_excludes_own_label() -> None:
    """A primeira linha de treino nao pode carregar informacao do proprio alvo."""
    train = _frame(100)
    encoded = CategoricalEncoder().fit_transform_train(train)
    first = encoded.sort_values("Fim").iloc[0]
    assert float(first["Classe_target_enc"]) == 0.0


def test_one_hot_categories_are_fixed_by_training() -> None:
    """Categoria de Frota vista so no teste nao cria coluna nova."""
    train = _frame(200)
    later = _frame(20, start="2026-05-01")
    later["Frota"] = "FROTA_NOVA"

    encoder = CategoricalEncoder().fit(train)
    encoded = encoder.transform(later)

    assert "Frota_FROTA_NOVA" not in encoded.columns
    frota_columns = [c for c in encoded.columns if c.startswith("Frota_")]
    assert frota_columns == [f"Frota_{c}" for c in encoder.one_hot_columns["Frota"]]
    # linha de categoria desconhecida fica com todos os indicadores em falso
    assert not encoded[frota_columns].to_numpy().any()


def test_split_rejects_negative_embargo() -> None:
    with pytest.raises(ValueError, match="embargo"):
        temporal_train_val_test_split(_frame(), embargo_hours=-1)


def test_backtest_folds_also_respect_the_embargo() -> None:
    """Cada fold do backtest precisa do mesmo intervalo de guarda do split."""
    folds = make_backtest_folds(_frame(600), n_folds=3)

    assert len(folds) == 3
    horizon = pd.Timedelta(hours=LABEL_HORIZON_HOURS)
    for train, val, test in folds:
        assert val["Fim"].min() - train["Fim"].max() >= horizon
        assert test["Fim"].min() - val["Fim"].max() >= horizon


def test_train_encoding_matches_inference_structure() -> None:
    """Train/serve skew: o treino nao pode ter distribuicao estranha a inferencia.

    A media expansiva linha a linha produzia um valor distinto por linha no
    treino (258 mil no dataset real) contra poucos valores por classe na
    inferencia. O encoding out-of-fold em blocos temporais mantem os dois lados
    na mesma escala estrutural.
    """
    train = _frame(500)
    encoder = CategoricalEncoder()
    encoded_train = encoder.fit_transform_train(train)
    encoded_later = encoder.transform(_frame(100, start="2026-06-01"))

    distintos_treino = encoded_train["Classe_target_enc"].nunique()
    distintos_inferencia = encoded_later["Classe_target_enc"].nunique()

    # com 5 blocos e poucas classes, o treino fica na mesma ordem de grandeza
    assert distintos_treino <= 5 * max(distintos_inferencia, 1) + 1
    assert distintos_treino < len(train) / 10, "um valor por linha indica skew"


def test_out_of_fold_encoding_stays_causal() -> None:
    """Nenhum bloco pode enxergar rotulo do proprio bloco nem do futuro."""
    n = 400
    train = _frame(n)
    # alvo muda de regime na metade: se houvesse vazamento, a primeira metade
    # ja refletiria o comportamento da segunda
    train["target_4h"] = [False] * (n // 2) + [True] * (n // 2)

    encoded = CategoricalEncoder().fit_transform_train(train).sort_values("Fim")
    primeira_metade = encoded["Classe_target_enc"].iloc[: n // 2]

    # a primeira metade so pode ver historico de alvo zero
    assert primeira_metade.max() < 0.5, "encoding da primeira metade viu o futuro"


def test_first_block_carries_no_target_information() -> None:
    """Sem historico anterior, o primeiro bloco fica neutro.

    Usar a media global do treino aqui pareceria inofensivo, mas vaza: se o
    alvo muda de regime ao longo do periodo, essa media resume tambem o futuro.
    """
    train = _frame(300)
    encoded = CategoricalEncoder().fit_transform_train(train).sort_values("Fim")
    primeiro_bloco = encoded["Classe_target_enc"].iloc[:60]
    assert primeiro_bloco.nunique() == 1
    assert float(primeiro_bloco.iloc[0]) == 0.0
