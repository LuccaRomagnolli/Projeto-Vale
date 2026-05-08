<p align="center">
  <img src="../pictures/vale-logo-removebg-preview.png" alt="Vale" width="120"/>
</p>

<h1 align="center">Antecipação de Alertas Críticos em Frota de Mineração</h1>

<p align="center">
  Antecipacao de alertas criticos <strong>"Don't Go"</strong> em equipamentos de mineracao,<br>
  com foco em priorizacao operacional por <code>Tag</code>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20%E2%80%93%203.13-1D9E75?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Modelo-Selecao-EF9F27?style=flat-square"/>
  <img src="https://img.shields.io/badge/Janela-4h-085041?style=flat-square"/>
  <img src="https://img.shields.io/badge/Split-70/15/15-888780?style=flat-square"/>
</p>

---

# Etapa 8 - Registro legado de otimizacao

Data: 04/05/2026

## Objetivo

Registrar que a antiga otimizacao especifica de uma familia foi substituida
pela selecao robusta multfamilia em `src/models/model_selection.py`.

## Entregaveis

| Entrega | Caminho |
|---|---|
| Selecao robusta vigente | `src/models/model_selection.py` |
| Gate de estabilidade | `src/models/stability_gate.py` |
| Artefato selecionado | `models/model_selected.joblib` |
| Relatorio JSON | `reports/model_selection_report.json` |
| Relatorio CSV | `reports/model_selection_report.csv` |
| Curva de threshold | `reports/model_selected_threshold_curve.csv` |
| Backtesting | `reports/model_selection_backtest_report.csv` |
| Testes | `tests/test_model_selection.py`, `tests/test_stability_gate.py` |

## Politica vigente

A rodada oficial executa `30` trials Optuna para cada candidato oficial:
`lightgbm_optuna`, `xgboost_optuna`, `hist_gbdt_optuna` e
`extra_trees_optuna`. A regressao logistica permanece apenas como baseline
diagnostico.

## Backtesting temporal

| Fold | Test Precision@15 | Test Recall@15 | Test Lift@15 |
|---:|---:|---:|---:|
| 1 | `0.6842` | `0.7500` | `2.1658` |
| 2 | `0.6246` | `0.7265` | `2.0470` |
| 3 | `0.6700` | `0.7283` | `2.0634` |

## Gate de estabilidade

| Metrica | Limite | Resultado |
|---|---:|---:|
| `std(test_top15_recall_at_k)` | `<= 0.03` | `0.0107` |
| `std(test_top15_precision_at_k)` | `<= 0.05` | `0.0254` |

## Decisao

Status: `SUBSTITUIDA`. O fluxo especifico de uma familia foi mantido apenas
como historico; o fluxo vigente e `make model-selection`.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
