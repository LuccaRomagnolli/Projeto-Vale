

# Mining Fleet Alert Anticipation

Antecipacao de alertas criticos **"Don't Go"** em equipamentos de mineracao,  
com foco em priorizacao operacional por `Tag`.



---

# Politica unica de promocao de modelo

Data: 04/05/2026

## Objetivo

Padronizar a decisao de promocao para evitar ambiguidade entre benchmark,
tuning e artefato operacional vigente.

## Regra obrigatoria

Um modelo so pode ser promovido quando todas as condicoes abaixo forem verdadeiras:


| Condicao                  | Regra                                                                |
| ------------------------- | -------------------------------------------------------------------- |
| Selecao                   | Benchmark robusto com 4 modelos, sem principal a priori              |
| Metrica primaria          | Maior `val_top15_recall_at_k`                                        |
| Desempates                | `val_top15_precision_at_k`, `val_top15_lift_vs_random`, `val_auc_pr` |
| Precision minima no teste | `Precision@Top15 Tag-dia >= 0.60`                                    |
| Recall minimo no teste    | `Recall@Top15 Tag-dia >= 0.70`                                       |
| Lift minimo no teste      | `Lift@Top15 Tag-dia >= 1.90`                                         |
| Estabilidade              | `make gate-stability` aprovado                                       |
| Segmentos raros           | Marcados como inconclusivos, nao como falha global automatica        |


## Gate de estabilidade


| Metrica                          | Limite    |
| -------------------------------- | --------- |
| `std(test_top15_recall_at_k)`    | `<= 0.03` |
| `std(test_top15_precision_at_k)` | `<= 0.05` |


## Fluxo oficial

```bash
make benchmark
make tune-hist-gbdt
make gate-stability
make evaluate
make evaluate-segments
make infer
```

## Artefatos vigentes


| Campo                        | Valor                                    |
| ---------------------------- | ---------------------------------------- |
| Modelo benchmark selecionado | `model_benchmark_selected`               |
| Artefato benchmark           | `models/model_benchmark_selected.joblib` |
| Modelo operacional vigente   | `hist_gbdt_tuned`                        |
| Artefato                     | `models/hist_gbdt_tuned.joblib`          |
| Threshold                    | `0.141388104973226`                      |
| Test Precision@15            | `0.6800`                                 |
| Test Recall@15               | `0.7409`                                 |
| Test Lift@15                 | `2.0910`                                 |


## Evidencias minimas

- `reports/model_benchmark_report.json`
- `reports/hist_gbdt_tuning_report.json`
- `reports/hist_gbdt_backtest_report.csv`
- `reports/operational_metrics_report.json`
- `reports/segment_operational_report.json`

## Decisao

Status: `VIGENTE`. O projeto esta metodologicamente apto para piloto operacional
assistido, com monitoramento continuo de drift, volume de alertas e segmentos
raros.

---

Vale · Mining Operations · Fleet Alert Anticipation