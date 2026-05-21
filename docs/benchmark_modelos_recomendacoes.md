<p align="center">
  <img src="../pictures/vale-logo-removebg-preview.png" alt="Vale" width="120"/>
</p>

<h1 align="center">Antecipação de Alertas Críticos em Frota de Mineração/h1>

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

# Benchmark de modelos e recomendacoes

Data: 04/05/2026

## Objetivo

Comparar candidatos supervisionados sob a mesma metodologia operacional:
validacao temporal, threshold calibrado fora do teste e selecao por
`val_top15_recall_at_k`, com desempates por `val_top15_precision_at_k`,
`val_top15_lift_vs_random` e `val_auc_pr`.

## Entregaveis

| Entrega | Caminho |
|---|---|
| Selecao robusta | `src/models/model_selection.py` |
| Relatorio JSON | `reports/model_selection/model_selection_report.json` |
| Relatorio CSV | `reports/model_selection/model_selection_report.csv` |
| Trials | `reports/model_selection/model_selection_trials.csv` |
| Scores | `reports/model_selection/model_selection_scores.parquet` |
| Artefato selecionado | `models/model_selected.joblib` |
| Testes | `tests/test_model_selection.py` |

## Modelos avaliados

| Modelo | Estimador | Papel |
|---|---|---|
| `lightgbm_optuna` | `LGBMClassifier` | Candidato oficial |
| `xgboost_optuna` | `XGBClassifier` | Candidato oficial |
| `hist_gbdt_optuna` | `HistGradientBoostingClassifier` | Candidato oficial |
| `logistic_regression_baseline` | `Pipeline` | Baseline diagnostico fora da disputa |

## Resultado da selecao

Os resultados vigentes devem ser lidos diretamente em
`reports/model_selection/model_selection_report.json` e `reports/model_selection/model_selection_report.csv`,
pois o artefato final depende dos trials Optuna executados em cada rodada.

## Recomendacoes

1. Manter `Top15 Tag-dia` como criterio executivo principal.
2. Usar AUC-PR e metricas ciclo-a-ciclo como diagnostico, nao como criterio
   isolado de promocao.
3. Promover somente apos selecao robusta, backtesting, gate de estabilidade e
   analise segmentada.

## Decisao

Status: `VIGENTE`. O benchmark simples e o tuning especifico foram substituidos
por selecao robusta multfamilia; o artefato operacional final e
`models/model_selected.joblib`.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
