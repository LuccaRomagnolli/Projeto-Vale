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

# Etapa 5 - Feature engineering sem leakage

Data: 04/05/2026

## Objetivo

Construir um dataset de features reutilizavel para modelagem, usando apenas
informacoes disponiveis ate o momento do ciclo e evitando vazamento temporal.

## Entregaveis

| Entrega | Caminho |
|---|---|
| Pipeline de features | `src/features/build_features.py` |
| Dataset de features | `data/processed/features/features_dataset.parquet` |
| Relatorio JSON | `data/processed/features/feature_report.json` |
| Splits temporais | `data/processed/features/splits/` |
| Metadados do split | `data/processed/features/splits/split_metadata.json` |
| Testes | `tests/test_build_features.py`, `tests/test_feature_engineering.py` |

## Familias de features

| Familia | Exemplos |
|---|---|
| Temporais | `hora_do_dia`, `dia_da_semana`, `mes`, `turno`, `is_fim_de_semana` |
| Duracao | `duracao_ciclo_min`, medias e desvios por janela |
| Rolling por Tag | `n_ciclos_Xh`, `n_alertas_Xh`, `freq_classe_atividade_Xh` |
| Intensidade | `alertas_por_hora_Xh`, razoes e deltas 4h contra 24h |
| Historico | `dias_desde_ultimo_alerta`, `Tag_freq`, `Operador_freq` |
| Categoricas | one-hot de `Frota` e `Tipo`, encoding historico de `Classe` |

## Garantia anti-leakage

As janelas historicas sao calculadas com informacao anterior ao registro, e as
colunas de futuro (`next_critical_event_time`, `tte_horas`, `target_4h`) nao
entram como features modelaveis.

## Resultado da execucao real

| Indicador | Valor |
|---|---:|
| Linhas | `377907` |
| Colunas totais | `57` |
| Features modelaveis | `48` |
| Positivos `target_4h` | `70811` |
| Taxa de positivos | `18.737679%` |

## Nulos relevantes

| Coluna | Percentual |
|---|---:|
| `next_critical_event_time` | `19.316128%` |
| `tte_horas` | `19.316128%` |
| `dias_desde_ultimo_alerta` | `10.323175%` |

## Decisao

Status: `CONCLUIDA`. O dataset de features esta pronto para split temporal,
baseline, modelos supervisionados e inferencia operacional.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
