<p align="center">
  <img src="../pictures/vale-logo-removebg-preview.png" alt="Vale" width="120"/>
</p>

<h1 align="center">Mining Fleet Alert Anticipation</h1>

<p align="center">
  Antecipacao de alertas criticos <strong>"Don't Go"</strong> em equipamentos de mineracao,<br>
  com foco em priorizacao operacional por <code>Tag</code>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20%E2%80%93%203.13-1D9E75?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Modelo-HistGBDT-EF9F27?style=flat-square"/>
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
| Benchmark | `src/models/benchmark_models.py` |
| Relatorio JSON | `reports/model_benchmark_report.json` |
| Relatorio CSV | `reports/model_benchmark_report.csv` |
| Scores | `reports/model_benchmark_scores.parquet` |
| Artefato vencedor | `models/model_benchmark_winner.joblib` |
| Testes | `tests/test_benchmark_models.py` |

## Modelos avaliados

| Modelo | Estimador | Papel |
|---|---|---|
| `hist_gbdt_balanced` | `HistGradientBoostingClassifier` | Campeao do benchmark |
| `hist_gbdt_regularized` | `HistGradientBoostingClassifier` | Alternativa regularizada |
| `lightgbm_balanced` | `LGBMClassifier` | Gradient boosting tabular |
| `logistic_regression_balanced` | `Pipeline` | Baseline linear supervisionado |
| `extra_trees_balanced` | `Pipeline` | Ensemble de arvores |
| `random_forest_balanced` | `Pipeline` | Referencia robusta |

## Resultado do benchmark

| Modelo | Val Precision@15 | Val Recall@15 | Val Lift@15 | Test Precision@15 | Test Recall@15 |
|---|---:|---:|---:|---:|---:|
| `hist_gbdt_balanced` | `0.6619` | `0.7493` | `2.1231` | `0.6822` | `0.7433` |
| `hist_gbdt_regularized` | `0.6548` | `0.7412` | `2.1002` | `0.6689` | `0.7288` |
| `lightgbm_balanced` | `0.6524` | `0.7385` | `2.0925` | `0.6756` | `0.7361` |
| `logistic_regression_balanced` | `0.6238` | `0.7062` | `2.0009` | `0.6733` | `0.7337` |
| `extra_trees_balanced` | `0.6238` | `0.7062` | `2.0009` | `0.6578` | `0.7167` |
| `random_forest_balanced` | `0.6214` | `0.7035` | `1.9933` | `0.6689` | `0.7288` |

## Vencedor

| Campo | Valor |
|---|---|
| Modelo | `hist_gbdt_balanced` |
| Threshold | `0.11401265843587063` |
| Test Precision@15 | `0.6822` |
| Test Recall@15 | `0.7433` |
| Test Lift@15 | `2.0979` |

## Recomendacoes

1. Manter `Top15 Tag-dia` como criterio executivo principal.
2. Usar AUC-PR e metricas ciclo-a-ciclo como diagnostico, nao como criterio
   isolado de promocao.
3. Promover somente apos tuning, backtesting, gate de estabilidade e analise
   segmentada.

## Decisao

Status: `CONCLUIDA`. O benchmark selecionou `hist_gbdt_balanced`; o artefato
operacional final foi refinado na etapa de tuning como `hist_gbdt_tuned`.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
