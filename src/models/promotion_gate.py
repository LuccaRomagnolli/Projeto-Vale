"""Politica de promocao de modelo como codigo executavel.

`docs/politica_promocao_modelo.md` define pisos obrigatorios de desempenho no
teste, mas nenhum deles era verificado programaticamente: apenas os dois desvios
padrao do gate de estabilidade tinham implementacao. Na pratica, um modelo com
`precision@15 = 0.45` seria promovido sem qualquer obstaculo -- o
`model_selection` grava o artefato antes de qualquer conferencia.

Este modulo transforma a tabela da politica em criterios avaliados um a um, com
veredito estruturado e saida de processo diferente de zero quando reprovado, de
modo que o CI barre a promocao.

Ao alterar um limite aqui, atualize `docs/politica_promocao_modelo.md` na mesma
mudanca -- os dois precisam contar a mesma historia.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.operational_scorecard import PRIMARY_TOP_K
from src.models.stability_gate import BACKTEST_REPORT_CSV, run_stability_gate
from src.utils.config import REPORTS_MODEL_SELECTION_DIR

SELECTION_REPORT_JSON = REPORTS_MODEL_SELECTION_DIR / "model_selection_report.json"
PROMOTION_REPORT_JSON = REPORTS_MODEL_SELECTION_DIR / "promotion_gate_report.json"

# Espelham docs/politica_promocao_modelo.md, secao "Regra obrigatoria".
MIN_TEST_PRECISION_AT_K = 0.60
MIN_TEST_RECALL_AT_K = 0.70
MIN_TEST_LIFT_VS_RANDOM = 1.90


@dataclass(frozen=True)
class Criterion:
    """Um criterio da politica, com o valor observado e o veredito."""

    name: str
    observed: float
    limit: float
    comparison: str  # ">=" ou "<="
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "criterio": self.name,
            "observado": self.observed,
            "limite": self.limit,
            "comparacao": self.comparison,
            "aprovado": self.passed,
        }


def _at_least(name: str, observed: float, limit: float) -> Criterion:
    return Criterion(name, float(observed), float(limit), ">=", float(observed) >= float(limit))


def _at_most(name: str, observed: float, limit: float) -> Criterion:
    return Criterion(name, float(observed), float(limit), "<=", float(observed) <= float(limit))


def load_selected_metrics(report_path: Path = SELECTION_REPORT_JSON) -> dict[str, Any]:
    """Le as metricas do modelo promovido no relatorio de selecao."""
    if not report_path.exists():
        raise FileNotFoundError(
            f"Relatorio de selecao ausente: {report_path}. "
            "Execute `python tasks.py model-selection` antes do gate."
        )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_model")
    if not isinstance(selected, dict) or not selected.get("model_name"):
        raise ValueError(f"Relatorio sem `selected_model.model_name`: {report_path}")
    return selected


def evaluate_performance_criteria(selected: dict[str, Any]) -> list[Criterion]:
    """Aplica os pisos de desempenho no conjunto de teste."""
    required = {
        f"test_top{PRIMARY_TOP_K}_precision_at_k": MIN_TEST_PRECISION_AT_K,
        f"test_top{PRIMARY_TOP_K}_recall_at_k": MIN_TEST_RECALL_AT_K,
        f"test_top{PRIMARY_TOP_K}_lift_vs_random": MIN_TEST_LIFT_VS_RANDOM,
    }
    missing = [key for key in required if key not in selected]
    if missing:
        raise ValueError(
            f"Relatorio de selecao nao expoe as metricas exigidas pela politica: {missing}. "
            "Sem elas o gate nao pode decidir e a promocao deve ser bloqueada."
        )
    return [_at_least(key, selected[key], limit) for key, limit in required.items()]


def run_promotion_gate(
    report_path: Path = SELECTION_REPORT_JSON,
    backtest_path: Path = BACKTEST_REPORT_CSV,
    output_path: Path = PROMOTION_REPORT_JSON,
) -> dict[str, Any]:
    """Avalia a politica completa e persiste o veredito."""
    selected = load_selected_metrics(report_path)
    criteria = evaluate_performance_criteria(selected)

    stability = run_stability_gate(backtest_path=backtest_path)
    criteria.append(
        _at_most("std_test_recall_at_k", stability["recall_std"], stability["max_recall_std"])
    )
    criteria.append(
        _at_most(
            "std_test_precision_at_k",
            stability["precision_std"],
            stability["max_precision_std"],
        )
    )

    failed = [c.name for c in criteria if not c.passed]
    verdict: dict[str, Any] = {
        "aprovado": not failed,
        "model_name": selected.get("model_name"),
        "top_k": PRIMARY_TOP_K,
        "criterios_reprovados": failed,
        "criterios": [c.as_dict() for c in criteria],
        "estabilidade": {
            "folds": stability["folds"],
            "metric_family": stability["metric_family"],
        },
        "politica": "docs/politica_promocao_modelo.md",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
    verdict["report_path"] = str(output_path)
    return verdict


def main() -> None:
    # Caminhos resolvidos aqui, e nao via default da assinatura: defaults sao
    # avaliados na definicao da funcao, entao um teste que redireciona os
    # caminhos do modulo continuaria escrevendo nos artefatos de producao.
    verdict = run_promotion_gate(
        report_path=SELECTION_REPORT_JSON,
        backtest_path=BACKTEST_REPORT_CSV,
        output_path=PROMOTION_REPORT_JSON,
    )
    print(f"[OK] Politica de promocao: {verdict['politica']}")
    print(f"[OK] Modelo avaliado: {verdict['model_name']}")
    for criterion in verdict["criterios"]:
        mark = "OK" if criterion["aprovado"] else "ERROR"
        print(
            f"[{mark}] {criterion['criterio']}: "
            f"{criterion['observado']:.4f} {criterion['comparacao']} {criterion['limite']:.4f}"
        )
    print(f"[OK] Veredito registrado em: {verdict['report_path']}")
    if not verdict["aprovado"]:
        raise SystemExit(
            f"[ERROR] Promocao bloqueada. Criterios reprovados: {verdict['criterios_reprovados']}"
        )


if __name__ == "__main__":
    main()
