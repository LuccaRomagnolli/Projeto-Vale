

# Mining Fleet Alert Anticipation

Antecipacao de alertas criticos **"Don't Go"** em equipamentos de mineracao,  
com foco em priorizacao operacional por `Tag`.



---

# Benchmark de modelos e recomendacoes

Data: 04/05/2026

## Objetivo

Comparar candidatos supervisionados sob a mesma metodologia operacional:
validacao temporal, threshold calibrado fora do teste e selecao por
`val_top15_recall_at_k`, com desempates por `val_top15_precision_at_k`,
`val_top15_lift_vs_random` e `val_auc_pr`.

## Entregaveis


| Entrega              | Caminho                                  |
| -------------------- | ---------------------------------------- |
| Benchmark            | `src/models/benchmark_models.py`         |
| Relatorio JSON       | `reports/model_benchmark_report.json`    |
| Relatorio CSV        | `reports/model_benchmark_report.csv`     |
| Scores               | `reports/model_benchmark_scores.parquet` |
| Artefato selecionado | `models/model_benchmark_selected.joblib` |
| Testes               | `tests/test_benchmark_models.py`         |


## Modelos avaliados


| Modelo                         | Estimador                        | Papel                                      |
| ------------------------------ | -------------------------------- | ------------------------------------------ |
| `lightgbm_balanced`            | `LGBMClassifier`                 | Boosting com alta capacidade tabular       |
| `hist_gbdt_regularized`        | `HistGradientBoostingClassifier` | Boosting regularizado e estavel            |
| `extra_trees_balanced`         | `Pipeline`                       | Ensemble de arvores com baixa variancia    |
| `logistic_regression_balanced` | `Pipeline`                       | Baseline linear calibravel e interpretavel |


## Resultado do benchmark


| Modelo                         | Val Precision@15 | Val Recall@15 | Val Lift@15 | Test Precision@15 | Test Recall@15 |
| ------------------------------ | ---------------- | ------------- | ----------- | ----------------- | -------------- |
| `hist_gbdt_balanced`           | `0.6619`         | `0.7493`      | `2.1231`    | `0.6822`          | `0.7433`       |
| `hist_gbdt_regularized`        | `0.6548`         | `0.7412`      | `2.1002`    | `0.6689`          | `0.7288`       |
| `lightgbm_balanced`            | `0.6524`         | `0.7385`      | `2.0925`    | `0.6756`          | `0.7361`       |
| `logistic_regression_balanced` | `0.6238`         | `0.7062`      | `2.0009`    | `0.6733`          | `0.7337`       |
| `extra_trees_balanced`         | `0.6238`         | `0.7062`      | `2.0009`    | `0.6578`          | `0.7167`       |
| `random_forest_balanced`       | `0.6214`         | `0.7035`      | `1.9933`    | `0.6689`          | `0.7288`       |


## Modelo selecionado


| Campo             | Valor                 |
| ----------------- | --------------------- |
| Modelo            | `hist_gbdt_balanced`  |
| Threshold         | `0.11401265843587063` |
| Test Precision@15 | `0.6822`              |
| Test Recall@15    | `0.7433`              |
| Test Lift@15      | `2.0979`              |


## Recomendacoes

1. Manter `Top15 Tag-dia` como criterio executivo principal.
2. Usar AUC-PR e metricas ciclo-a-ciclo como diagnostico, nao como criterio
  isolado de promocao.
3. Promover somente apos tuning, backtesting, gate de estabilidade e analise
  segmentada.

## Decisao

Status: `CONCLUIDA`. O benchmark robusto seleciona um modelo apenas apos
comparar 4 candidatos em multiplas iteracoes; o artefato operacional final
permanece refinado na etapa de tuning (`hist_gbdt_tuned`).

---

Vale · Mining Operations · Fleet Alert Anticipation