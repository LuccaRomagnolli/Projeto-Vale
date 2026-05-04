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

# Politica unica de promocao de modelo

Data: 04/05/2026

## Objetivo

Padronizar a decisao de promocao para evitar ambiguidade entre modelo
principal, campeao de benchmark e artefato operacional vigente.

## Regra obrigatoria

Um modelo so pode ser promovido quando todas as condicoes abaixo forem verdadeiras:

| Condicao | Regra |
|---|---|
| Selecao | Campeao por validacao temporal operacional |
| Metrica primaria | Maior `val_top15_recall_at_k` |
| Desempates | `val_top15_precision_at_k`, `val_top15_lift_vs_random`, `val_auc_pr` |
| Precision minima no teste | `Precision@Top15 Tag-dia >= 0.60` |
| Recall minimo no teste | `Recall@Top15 Tag-dia >= 0.70` |
| Lift minimo no teste | `Lift@Top15 Tag-dia >= 1.90` |
| Estabilidade | `make gate-stability` aprovado |
| Segmentos raros | Marcados como inconclusivos, nao como falha global automatica |

## Gate de estabilidade

| Metrica | Limite |
|---|---:|
| `std(test_top15_recall_at_k)` | `<= 0.03` |
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

## Artefato operacional vigente

| Campo | Valor |
|---|---|
| Modelo | `hist_gbdt_tuned` |
| Artefato | `models/hist_gbdt_tuned.joblib` |
| Threshold | `0.141388104973226` |
| Test Precision@15 | `0.6800` |
| Test Recall@15 | `0.7409` |
| Test Lift@15 | `2.0910` |

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

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
