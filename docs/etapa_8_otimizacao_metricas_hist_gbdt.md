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

# Etapa 8 - Otimizacao de metricas HistGBDT

Data: 04/05/2026

## Objetivo

Otimizar o candidato HistGradientBoosting para maximizar utilidade operacional
em `Top15 Tag-dia`, mantendo calibracao honesta em validacao temporal e
backtesting para estabilidade.

## Entregaveis

| Entrega | Caminho |
|---|---|
| Tuning e backtesting | `src/models/tune_hist_gbdt.py` |
| Gate de estabilidade | `src/models/stability_gate.py` |
| Artefato promovido | `models/hist_gbdt_tuned.joblib` |
| Relatorio JSON | `reports/hist_gbdt_tuning_report.json` |
| Relatorio CSV | `reports/hist_gbdt_tuning_report.csv` |
| Curva de threshold | `reports/hist_gbdt_threshold_curve.csv` |
| Backtesting | `reports/hist_gbdt_backtest_report.csv` |
| Testes | `tests/test_tune_hist_gbdt.py`, `tests/test_stability_gate.py` |

## Melhor candidato

| Campo | Valor |
|---|---|
| Candidato | `hist_gbdt_tuned_04` |
| Threshold | `0.141388104973226` |
| `learning_rate` | `0.04` |
| `max_iter` | `420` |
| `max_leaf_nodes` | `21` |
| `min_samples_leaf` | `120` |
| `l2_regularization` | `2.0` |
| `class_weight` | `balanced` |
| Features | `48` |

## Resultado do candidato promovido

| Split | Precision@15 | Recall@15 | Lift@15 | AUC-PR | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Treino | `0.8413` | `0.7779` | `2.2497` | `0.6805` | `0.8992` |
| Validacao | `0.6643` | `0.7520` | `2.1307` | `0.2450` | `0.7408` |
| Teste | `0.6800` | `0.7409` | `2.0910` | `0.2589` | `0.7371` |

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

Status: `CONCLUIDA`. O candidato `hist_gbdt_tuned_04` atende as metas
operacionais, passa no gate de estabilidade e e o artefato operacional vigente.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
