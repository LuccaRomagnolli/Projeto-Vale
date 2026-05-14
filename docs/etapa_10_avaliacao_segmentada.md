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

# Etapa 10 - Avaliacao segmentada operacional

Data: 04/05/2026

## Objetivo

Identificar onde o modelo e confiavel, onde exige cuidado e quais segmentos ou
Tags devem receber trilha dedicada antes de ampliacao operacional.

## Entregaveis

| Entrega | Caminho |
|---|---|
| Analise segmentada | `src/evaluation/segment_analysis.py` |
| Relatorio JSON | `reports/segments/segment_operational_report.json` |
| Threshold por segmento | `reports/segments/segment_threshold_metrics.csv` |
| TopK por segmento | `reports/segments/segment_topk_tag_day_metrics.csv` |
| Hotspots por Tag | `reports/segments/segment_tag_hotspots.csv` |
| Testes | `tests/test_segment_analysis.py` |

## Segmentos avaliados

| Segmento | Uso |
|---|---|
| `Frota` | Risco por familia de frota |
| `Tipo` | Caminhao versus escavadeira |
| `turno` | Manha, tarde e noite |
| `Classe` | Estado operacional do ciclo |
| `Tag` | Hotspots e falsos negativos recorrentes |

## Segmentos Top15 destacados

| Segmento | Valor | Precision@15 | Recall@15 | Lift | Status |
|---|---|---:|---:|---:|---|
| `turno` | `manha` | `0.4690` | `0.8312` | `2.3452` | `ok` |
| `turno` | `tarde` | `0.4368` | `0.7755` | `2.1875` | `ok` |
| `turno` | `noite` | `0.5200` | `0.7290` | `2.0525` | `ok` |
| `Tipo` | `Caminhao` | `0.6756` | `0.7397` | `1.6420` | `ok` |
| `Classe` | `Parado` | `0.6511` | `0.7834` | `1.3649` | `ok` |
| `Classe` | `Hibernando` | `0.0000` | `0.0000` | `0.0000` | `inconclusivo_baixa_prevalencia` |
| `Tipo` | `Escavadeira` | `0.0074` | `1.0000` | `1.0000` | `inconclusivo_baixa_prevalencia` |
| `Frota` | `LeTourneau L 1850` | `0.0074` | `1.0000` | `1.0000` | `inconclusivo_baixa_prevalencia` |

## Hotspots de atencao

| Tag | Dias positivos | Dias selecionados | Positivos perdidos | Leitura |
|---|---:|---:|---:|---|
| `CA65921` | `25` | `0` | `25` | Alto risco nao priorizado |
| `CA65789` | `14` | `0` | `14` | Requer investigacao |
| `CA65790` | `13` | `0` | `13` | Requer investigacao |
| `CA65937` | `15` | `4` | `11` | Seleciona pouco, mas com precisao |

## Decisao

Status: `CONCLUIDA`. O modelo e viavel globalmente para TopK, mas segmentos de
baixa prevalencia e Tags com perda recorrente devem ter trilha dedicada de
monitoramento, coleta ou calibracao local.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
