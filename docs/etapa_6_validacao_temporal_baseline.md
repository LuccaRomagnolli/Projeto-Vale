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

# Etapa 6 - Validacao temporal e baseline

Data: 04/05/2026

## Objetivo

Congelar um split temporal honesto e treinar um baseline heuristico para servir
como referencia minima antes da selecao de modelos supervisionados.

## Entregaveis

| Entrega | Caminho |
|---|---|
| Split temporal | `src/models/validation.py` |
| Baseline | `src/models/train_baseline.py` |
| Artefato baseline | `models/baseline_heuristico.joblib` |
| Relatorio baseline | `reports/baseline_report.json` |
| Scores baseline | `reports/baseline_scores.parquet` |
| Metadados do split | `data/processed/features/splits/split_metadata.json` |
| Testes | `tests/test_validation.py`, `tests/test_train_baseline.py` |

## Split temporal

| Split | Linhas | Periodo |
|---|---:|---|
| Treino | `264534` | `2025-01-01 03:03:43+00:00` a `2025-05-06 06:00:00+00:00` |
| Validacao | `56686` | `2025-05-06 06:00:00+00:00` a `2025-06-02 15:00:00+00:00` |
| Teste | `56687` | `2025-06-02 15:00:00+00:00` a `2025-07-01 03:00:00+00:00` |

## Baseline heuristico

| Item | Valor |
|---|---|
| Modelo | `baseline_heuristico_24h` |
| Score | `n_alertas_24h / 24`, limitado a `[0, 1]` |
| Threshold | `0.0` |

## Metricas do baseline

| Split | Recall | Precision | F1 | AUC-PR | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Treino | `1.0000` | `0.2081` | `0.3446` | `0.2081` | `0.5000` |
| Validacao | `1.0000` | `0.1281` | `0.2271` | `0.1281` | `0.5000` |
| Teste | `1.0000` | `0.1498` | `0.2605` | `0.1498` | `0.5000` |

## Decisao

Status: `CONCLUIDA`. O baseline captura todos os positivos porque alerta de
forma ampla, mas tem baixa precisao e nao discrimina risco; por isso a etapa
seguinte deve usar modelo supervisionado e ranking operacional.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
