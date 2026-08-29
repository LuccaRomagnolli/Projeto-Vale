"""Calibracao de probabilidade do score operacional.

Os candidatos oficiais treinam com `scale_pos_weight` ou
`class_weight="balanced"` para lidar com o desbalanceamento. Isso melhora a
capacidade de ordenar, mas distorce a escala: o valor devolvido por
`predict_proba` deixa de ser uma probabilidade e passa a ser apenas um score
monotono. Na interface, "score 0,70" era lido pela operacao como "70% de chance
de alerta", o que nao correspondia a frequencia observada.

O que a calibracao muda e o que nao muda:

- **Nao muda** as metricas TopK Tag-dia. A calibracao isotonica e monotona, e
  ranking e invariante a transformacao monotona -- `precision@15`, `recall@15` e
  `lift@15` permanecem identicos por construcao.
- **Muda** a leitura do score e, por consequencia, o valor numerico do threshold
  calibrado, que passa a viver na escala de probabilidade.

Por isso a calibracao e opcional e desligada por padrao: liga-la altera o
threshold publicado e exige revisar a documentacao operacional.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


class ProbabilityCalibrator:
    """Mapeia scores brutos para probabilidades usando regressao isotonica.

    O ajuste usa exclusivamente a validacao, o mesmo conjunto onde o threshold
    e calibrado -- nunca o teste.
    """

    def __init__(self) -> None:
        self._isotonic: IsotonicRegression | None = None
        self.fitted: bool = False

    def fit(self, y_true: pd.Series | np.ndarray, y_score: pd.Series | np.ndarray) -> Any:
        y_true_arr = np.asarray(y_true).astype(int)
        y_score_arr = np.asarray(y_score).astype(float)
        if y_score_arr.size == 0:
            raise ValueError("Nao ha scores para calibrar.")
        if len(np.unique(y_true_arr)) < 2:
            raise ValueError(
                "Calibracao exige as duas classes na validacao; "
                "com uma unica classe o mapeamento seria degenerado."
            )
        self._isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._isotonic.fit(y_score_arr, y_true_arr)
        self.fitted = True
        return self

    def transform(self, y_score: pd.Series | np.ndarray) -> np.ndarray:
        if not self.fitted or self._isotonic is None:
            raise ValueError("Calibrador nao ajustado: chame fit() antes de transform().")
        return np.asarray(self._isotonic.predict(np.asarray(y_score).astype(float)))


def calibration_error(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error: distancia media entre score previsto e frequencia real.

    Serve para dizer se a calibracao vale a pena e para comprovar que ela
    melhorou a leitura do score.
    """
    y_true_arr = np.asarray(y_true).astype(float)
    y_score_arr = np.asarray(y_score).astype(float)
    if y_score_arr.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total_error = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (y_score_arr >= low) & (y_score_arr < high if high < 1.0 else y_score_arr <= high)
        if not in_bin.any():
            continue
        weight = in_bin.mean()
        total_error += weight * abs(y_score_arr[in_bin].mean() - y_true_arr[in_bin].mean())
    return float(total_error)
