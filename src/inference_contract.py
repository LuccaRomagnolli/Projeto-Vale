"""Contrato de entrada da inferencia operacional.

Ate aqui o pipeline preenchia com `0.0` qualquer feature ausente no lote e
apenas registrava o fato em metadados. Num piloto diario isso e a falha mais
perigosa possivel: um lote com schema alterado e pontuado normalmente e produz
um ranking de inspecao sem sentido, sem erro e sem sinal visivel.

A validacao tem duas camadas, com papeis distintos:

1. Estrutural -- derivada do proprio artefato promovido (`feature_columns`).
   E a autoridade sobre o que o modelo espera e, por vir do artefato, nao pode
   divergir dele quando o modelo for retreinado.
2. De valores -- schema pandera sobre as colunas cruas do lote, cobrindo o que
   a checagem estrutural nao alcanca: nulos em campos de identidade, datas
   improprias e faixas numericas impossiveis.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pandera.pandas as pa

# Colunas de identidade e contexto que todo lote operacional precisa ter para
# que o ranking Tag-dia seja construivel.
REQUIRED_CONTEXT_COLUMNS = ("Tag", "Fim")

# Colunas categoricas cruas consumidas pelo CategoricalEncoder.
RAW_CATEGORICAL_COLUMNS = ("Tag", "Classe", "Frota", "Tipo")


class InferenceContractError(ValueError):
    """Erro de contrato do lote de inferencia, com mensagem acionavel."""


def build_batch_schema() -> pa.DataFrameSchema:
    """Schema de valores para o lote cru, antes dos encodings."""
    return pa.DataFrameSchema(
        columns={
            "Tag": pa.Column(str, nullable=False, coerce=True),
            "Fim": pa.Column(
                "datetime64[ns, UTC]",
                nullable=False,
                coerce=True,
                checks=pa.Check(
                    lambda s: (s.dt.year >= 2000) & (s.dt.year <= 2100),
                    error="Fim fora da faixa plausivel (2000-2100)",
                ),
            ),
            "duracao_ciclo_min": pa.Column(
                float,
                nullable=True,
                required=False,
                coerce=True,
                checks=pa.Check.ge(0, error="duracao_ciclo_min negativa"),
            ),
            "n_alertas_4h": pa.Column(
                float,
                nullable=True,
                required=False,
                coerce=True,
                checks=pa.Check.ge(0, error="n_alertas_4h negativo"),
            ),
            "dias_desde_ultimo_alerta": pa.Column(
                float,
                nullable=True,
                required=False,
                coerce=True,
                checks=pa.Check.in_range(
                    0,
                    3650,
                    error=(
                        "dias_desde_ultimo_alerta fora da faixa plausivel (0-3650). "
                        "Valores na casa de milhares indicam mistura de resolucao "
                        "temporal entre as fontes; ver src/utils/timeutils.py"
                    ),
                ),
            ),
        },
        strict=False,  # colunas extras sao toleradas e ignoradas na pontuacao
        coerce=True,
    )


def validate_batch_values(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica o schema de valores, convertendo falhas em erro acionavel."""
    missing_context = [c for c in REQUIRED_CONTEXT_COLUMNS if c not in df.columns]
    if missing_context:
        raise InferenceContractError(
            f"Lote sem colunas de contexto obrigatorias: {missing_context}. "
            "Sem elas o ranking Tag-dia nao pode ser construido."
        )
    try:
        return build_batch_schema().validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        failures = exc.failure_cases[["column", "check", "failure_case"]].head(10)
        raise InferenceContractError(
            f"Lote reprovado na validacao de valores:\n{failures.to_string(index=False)}"
        ) from exc


def check_feature_coverage(
    df: pd.DataFrame,
    feature_columns: list[str],
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Compara as colunas do lote com o que o artefato espera.

    Por padrao falha alto: um lote incompleto nao deve ser pontuado. Passar
    `allow_missing=True` e uma decisao explicita de quem chama, nunca o padrao.
    """
    missing = [column for column in feature_columns if column not in df.columns]
    extra = [column for column in df.columns if column not in feature_columns]

    if missing and not allow_missing:
        raise InferenceContractError(
            f"Lote nao contem {len(missing)} das {len(feature_columns)} features "
            f"esperadas pelo artefato: {missing[:12]}"
            f"{' ...' if len(missing) > 12 else ''}. "
            "Preencher com zero produziria um ranking sem sentido. Verifique se o "
            "lote passou por `build_feature_dataset` e se o encoder categorico "
            "foi aplicado."
        )
    return {"missing_feature_columns": missing, "extra_columns_ignored": extra}
