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

# Etapa 4 - EDA orientada a decisao

Data: 04/05/2026

## Objetivo

Executar analise exploratoria reprodutivel para entender volume, cobertura
temporal, distribuicao do alvo, qualidade de dados e variaveis operacionais
antes da engenharia de atributos.

## Entregaveis

| Entrega | Caminho |
|---|---|
| Pipeline de EDA | `src/eda/run_eda.py` |
| Relatorio | `reports/eda_report.md` |
| Sumario visual | `reports/eda_executivo.png` |
| Figuras | `reports/figures/` |
| Testes | `tests/test_eda.py`, `tests/test_eda_pipeline_outputs.py` |

## Resultado principal

| Indicador | Valor |
|---|---:|
| Registros analisados | `377907` |
| Tags unicas | `47` |
| Frotas unicas | `5` |
| Tipos unicos | `2` |
| Positivos `target_4h` | `70811` |
| Taxa de positivos | `18.737679%` |
| Inicio minimo | `2025-01-01 03:00:00+00:00` |
| Inicio maximo | `2025-07-01 02:57:55+00:00` |
| Duracao media de ciclo | `29.6854 min` |
| Duracao mediana de ciclo | `22.9 min` |
| P95 de duracao | `60.0 min` |

## Qualidade observada

| Coluna | Nulos |
|---|---:|
| `next_critical_event_time` | `19.3161%` |
| `tte_horas` | `19.3161%` |
| Colunas obrigatorias de ingestao | `0.0%` |

## Figuras geradas

- `reports/figures/eda_target_distribution.png`
- `reports/figures/eda_ciclos_por_hora.png`
- `reports/figures/eda_top_frotas.png`
- `reports/figures/eda_duracao_ciclo_hist.png`
- `reports/figures/eda_top_classes.png`

## Decisao

Status: `CONCLUIDA`. O dataset tem volume, cobertura temporal e prevalencia
adequados para modelagem supervisionada com avaliacao temporal e metrica TopK.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
