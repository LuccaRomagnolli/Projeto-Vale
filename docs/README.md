<p align="center">
  <img src="../pictures/vale-logo-removebg-preview.png" alt="Vale" width="120"/>
</p>

# Documentacao do Projeto

Data de atualizacao: 04/05/2026

## Fonte oficial atual

Para status e decisoes atuais do projeto, priorize:

1. `README.md`
2. `docs/politica_promocao_modelo.md`
3. `docs/controle_alteracoes.md`
4. `docs/revisao_geral_documentacao_2026-05-04.md`

## Documentos por finalidade

- Fundacao e dados:
  - `docs/etapa_1_fundacao_engenharia.md`
  - `docs/etapa_2_ingestao_contrato_dados.md`
  - `docs/etapa_3_rotulacao_alert_labeler.md`
  - `docs/etapa_4_eda_orientada_decisao.md`
  - `docs/etapa_5_feature_engineering.md`
  - `docs/etapa_6_validacao_temporal_baseline.md`
- Modelagem e promocao:
  - `docs/etapa_7_modelo_principal.md`
  - `docs/etapa_8_otimizacao_metricas_hist_gbdt.md`
  - `docs/benchmark_modelos_recomendacoes.md`
  - `docs/politica_promocao_modelo.md`
  - `docs/controle_alteracoes.md`
- Operacao e risco:
  - `docs/etapa_9_metricas_operacionais_confiaveis.md`
  - `docs/etapa_10_avaliacao_segmentada.md`
  - `docs/relatorio_eficiencia_2026-05-03.md`
  - `docs/notebook_codigo_e_contexto_projeto.md`

## Regra de atualizacao

Quando alterar pipeline, comandos `make`, metricas oficiais ou artefato
operacional, atualize no mesmo PR:

- `README.md`
- `docs/README.md`
- documento de etapa afetado
- `docs/controle_alteracoes.md` (se houver mudanca metodologica)
