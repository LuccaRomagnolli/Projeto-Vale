from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FrequencyBaseline:
    """Baseline de frequencia historica por equipamento (Tag)."""

    tag_col: str = "Tag"
    global_rate_: float | None = None
    tag_rates_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FrequencyBaseline:
        if self.tag_col not in X.columns:
            raise ValueError(f"Coluna de tag nao encontrada: {self.tag_col}")

        y = pd.Series(y).astype(float)
        self.global_rate_ = float(y.mean()) if len(y) else 0.0

        tmp = pd.DataFrame({self.tag_col: X[self.tag_col], "target": y})
        self.tag_rates_ = tmp.groupby(self.tag_col)["target"].mean()
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if self.global_rate_ is None or self.tag_rates_ is None:
            raise RuntimeError("Modelo nao treinado. Execute fit() primeiro.")
        if self.tag_col not in X.columns:
            raise ValueError(f"Coluna de tag nao encontrada: {self.tag_col}")

        probs = X[self.tag_col].map(self.tag_rates_).fillna(self.global_rate_)
        return probs.clip(0, 1)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
