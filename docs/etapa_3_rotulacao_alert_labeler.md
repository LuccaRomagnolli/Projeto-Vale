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

# Etapa 3 - Rotulacao robusta com Alert Labeler

Data: 04/05/2026

## Objetivo

Construir o alvo supervisionado `target_4h`, indicando se um equipamento tera
evento critico futuro em ate 4 horas, e registrar `tte_horas` como tempo ate o
proximo evento critico.

## Entregaveis

| Entrega | Caminho |
|---|---|
| Rotulador | `src/alert_labeler.py` |
| Integracao no pipeline | `src/data/make_dataset.py` |
| Eventos criticos | `data/processed/labeled/critical_events.parquet` |
| Dataset rotulado | `data/processed/labeled/apontamentos_labeled.parquet` |
| Relatorio JSON | `data/processed/labeled/labeling_report.json` |
| Relatorio Markdown | `data/processed/labeled/labeling_report.md` |
| Testes | `tests/test_alert_labeler.py` |

## Metodo

| Fonte | Uso |
|---|---|
| Regras de negocio | Filtrar eventos `Muito Alto` e contexto critico |
| Telemetria mensal | Localizar eventos operacionais candidatos |
| `Is_Dont_Go` | Preservar casos ja marcados como criticos |
| `Tag` e tempo | Associar evento futuro ao ciclo operacional |

## Resultado da execucao real

| Indicador | Valor |
|---|---:|
| Arquivos de telemetria processados | `6` |
| Eventos analisados | `37164054` |
| Match completo por contexto | `0` |
| Match por evento | `105976` |
| Fallback `Is_Dont_Go=1` | `19962` |
| Eventos criticos finais | `107002` |
| Tags com sobreposicao apontamento x evento | `34` |
| Registros rotulados | `377907` |
| Positivos `target_4h` | `70811` |
| Taxa de positivos | `18.737679%` |
| Primeiro evento critico | `2025-01-01 03:37:35.267000+00:00` |
| Ultimo evento critico | `2025-07-01 02:59:08.113000+00:00` |

## Evidencias

- `data/processed/labeled/labeling_report.json`
- `data/processed/labeled/apontamentos_labeled.parquet`
- `make label`

## Decisao

Status: `CONCLUIDA`. A rotulacao gera prevalencia suficiente para treinamento
supervisionado e preserva rastreabilidade para auditoria.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
