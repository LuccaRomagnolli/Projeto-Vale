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

# Etapa 9 - Metricas operacionais confiaveis

Data: 04/05/2026

## Objetivo

Avaliar o modelo como ferramenta de priorizacao diaria de manutencao, usando
ranking `TopK Tag-dia` como metrica primaria, em vez de depender apenas de
metricas ciclo-a-ciclo.

## Entregaveis

| Entrega | Caminho |
|---|---|
| Avaliador operacional | `src/evaluation/evaluate_model.py` |
| Relatorio JSON | `reports/operational/operational_metrics_report.json` |
| Budget por percentual | `reports/operational/operational_budget_metrics.csv` |
| TopK diario | `reports/operational/operational_daily_topk_metrics.csv` |
| Alertas deduplicados | `reports/operational/operational_deduplicated_alerts.csv` |
| Testes | `tests/test_operational_evaluation.py` |

## Modelo avaliado

| Campo | Valor |
|---|---|
| Modelo | Lido de `models/model_selected.joblib` |
| Artefato | `models/model_selected.joblib` |
| Threshold | Calibrado na validacao e persistido no artefato |
| Features | `48` |
| Split analisado | Teste temporal |

## Metricas ciclo-a-ciclo no threshold

| Indicador | Valor |
|---|---:|
| Linhas no teste | `56687` |
| Prevalencia | `0.1498` |
| Preditos positivos | `24925` |
| Taxa alertada | `0.4397` |
| Precision | `0.2727` |
| Recall | `0.8006` |
| Lift vs aleatorio | `1.8208` |

## Metricas TopK Tag-dia no teste

| TopK Tags/dia | Alertas selecionados | Precision@K | Recall@K | Lift |
|---:|---:|---:|---:|---:|
| 3 | `90` | `0.6889` | `0.1501` | `2.1184` |
| 5 | `150` | `0.6400` | `0.2324` | `1.9680` |
| 10 | `300` | `0.6767` | `0.4915` | `2.0808` |
| 15 | `450` | `0.6756` | `0.7361` | `2.0774` |
| 20 | `600` | `0.6233` | `0.9056` | `1.9168` |

## Deduplicacao operacional

| Indicador | Valor |
|---|---:|
| Cooldown | `4h` |
| Preditos positivos brutos | `24925` |
| Alertas deduplicados | `2541` |
| Alertas deduplicados por dia | `87.62` |
| Precision deduplicada | `0.2621` |

## Decisao

Status: `CONCLUIDA`. O modo recomendado de uso e ranking `Top15 Tag-dia`, que
captura `306` de `413` dias-Tag positivos no teste e entrega lift maior que
aleatorio.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
