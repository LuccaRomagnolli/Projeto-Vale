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

# Documentacao do Projeto

Data de atualizacao: 04/05/2026

## Objetivo

Organizar a documentacao tecnica e executiva do projeto de antecipacao de
alertas criticos em frota de mineracao. Esta pasta detalha as etapas de dados,
rotulacao, exploracao, features, modelagem, avaliacao operacional, analise
segmentada, promocao e inferencia.

## Resultado oficial atual

| Item | Valor |
|---|---|
| Registros do dataset | `377907` |
| Tags unicas | `47` |
| Frotas unicas | `5` |
| Tipos unicos | `2` |
| Eventos criticos finais | `107002` |
| Positivos `target_4h` | `70811` (`18.737679%`) |
| Features totais | `57` colunas |
| Features modelaveis | `44` |
| Split temporal | `70/15/15` com embargo de `4h` |
| Seleção oficial | `lightgbm_optuna`, `xgboost_optuna`, `hist_gbdt_optuna` |
| Baseline diagnóstico | `logistic_regression_baseline` |
| Modelo selecionado | `hist_gbdt_optuna` |
| Artefato promovido | `models/model_selected.joblib` |
| Relatorio de seleção | `reports/model_selection/model_selection_report.json` |
| Ranking operacional | `reports/daily_priority_top15.csv` |
| Test `Precision@Top15 Tag-dia` | `0.8133` |
| Test `Recall@Top15 Tag-dia` | `0.8862` |
| Test `Lift@Top15 Tag-dia` | `2.5010` |

> Metricas atualizadas em 29/08/2026. Ver `docs/controle_alteracoes.md`,
> alteracao 04, para a decomposicao entre o reparo das features de alerta e a
> correcao de vazamento.

## Indice por etapa

| Etapa | Documento | Foco |
|---:|---|---|
| 1 | `docs/etapa_1_fundacao_engenharia.md` | Engenharia, qualidade e automacao |
| 2 | `docs/etapa_2_ingestao_contrato_dados.md` | Ingestao e contrato dos apontamentos |
| 3 | `docs/etapa_3_rotulacao_alert_labeler.md` | Rotulacao `target_4h` com telemetria critica |
| 4 | `docs/etapa_4_eda_orientada_decisao.md` | EDA orientada a decisao |
| 5 | `docs/etapa_5_feature_engineering.md` | Features sem leakage |
| 6 | `docs/etapa_6_validacao_temporal_baseline.md` | Split temporal e baseline |
| 7 | `docs/etapa_7_selecao_robusta_modelos.md` | Selecao robusta de modelos |
| 8 | `docs/etapa_8_otimizacao_metricas_hist_gbdt.md` | Registro legado de otimizacao substituida |
| 9 | `docs/etapa_9_metricas_operacionais_confiaveis.md` | Metricas operacionais TopK |
| 10 | `docs/etapa_10_avaliacao_segmentada.md` | Segmentos, riscos e hotspots |
| 11 | `docs/etapa_11_inferencia_operacional.md` | Contrato de inferencia operacional |

## Documentos de governanca

| Documento | Finalidade |
|---|---|
| `docs/benchmark_modelos_recomendacoes.md` | Selecao robusta de candidatos e recomendacoes |
| `docs/politica_promocao_modelo.md` | Regra unica para promocao de modelo |
| `docs/controle_alteracoes.md` | Registro das mudancas metodologicas |
| `docs/simulacao_piloto.md` | Simulacao economica e de manutencao (NAO e medicao) |
| `docs/notebook_codigo_e_contexto_projeto.md` | Contexto do notebook principal |
| `docs/relatorio_eficiencia_2026-05-03.md` | Revisao de eficiencia ponta a ponta |
| `docs/revisao_geral_documentacao_2026-05-04.md` | Revisao de consistencia documental |

## Fonte de verdade

Para status atual, priorize nesta ordem:

1. `README.md`
2. `docs/README.md`
3. `docs/politica_promocao_modelo.md`
4. `docs/controle_alteracoes.md`
5. Relatorios em `reports/` e `data/processed/`

## Organizacao de reports

Somente o artefato operacional mais importante fica na raiz:
`reports/daily_priority_top15.csv`.

As demais evidencias ficam em subpastas:

| Pasta | Conteudo |
|---|---|
| `reports/baseline/` | baseline heuristico e scores |
| `reports/eda/` | EDA, figuras e relatorio exploratorio |
| `reports/inference/` | scores granulares de inferencia |
| `reports/model_selection/` | selecao oficial, trials, backtest e importancia |
| `reports/modeling_legacy/` | artefatos historicos/legados de modelagem |
| `reports/operational/` | metricas operacionais TopK e budget |
| `reports/segments/` | avaliacao segmentada e hotspots |

## Organizacao de referencias

Materiais externos de apoio ficam em `docs/reference/`, mantendo a raiz do
repositorio focada em arquivos executaveis, configuracoes e documentacao
principal.

| Pasta | Conteudo |
|---|---|
| `docs/reference/` | guias, enunciados e materiais de referencia do projeto |

## Regra de manutencao

Quando pipeline, comandos `tasks.py`/`make`, metricas oficiais, artefato operacional ou
politica de promocao forem alterados, atualizar no mesmo PR:

1. `README.md`
2. `docs/README.md`
3. documento da etapa afetada
4. `docs/controle_alteracoes.md`, se houver mudanca metodologica

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
