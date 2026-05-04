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

# Etapa 3b - Rotulacao robusta (fonte expandida)

Data: 30/04/2026

## Objetivo

Construir a variavel alvo de risco na janela de predição de 4 horas por Tag do equipamento.

## Entregaveis implementados

1. Novo modulo `src/alert_labeler.py`.
2. Carga e filtro das regras de negocio criticas (`NIVEL = Muito Alto`) a partir de `CMA`.
3. Leitura de telemetria operacional mensal a partir de `data/raw/.../telemetria/telemetry_*.parquet`.
4. Match de eventos criticos:
   - match completo por `TIPO + EVENTO + SITUACAO + NIVEL` quando o evento traz contexto completo;
   - match por `EVENTO` quando contexto de situacao/nivel nao existe no evento bruto;
   - fallback por `Is_Dont_Go = 1` para preservar casos rotulados na telemetria.
5. Construcao de:
   - `target_4h`: 1 se existe evento critico futuro em ate 4 horas;
   - `tte_horas`: tempo (em horas) ate o proximo evento critico.
6. Persistencia de saidas:
   - `data/processed/labeled/critical_events.parquet`
   - `data/processed/labeled/apontamentos_labeled.parquet`
   - `data/processed/labeled/labeling_report.json`
   - `data/processed/labeled/labeling_report.md`
7. Integracao no pipeline `make label` via `src/data/make_dataset.py`.
8. Testes unitarios da etapa em `tests/test_alert_labeler.py`.

## Validacao planejada

1. `make lint`
2. `make test`
3. `make label`

## Resultado da validacao

- `make lint`: OK
- `make test`: OK
- `make label`: OK
- Testes: `23 passed`
- Cobertura total em `src`: `82%`
- Execucao real da rotulacao (3b):
  - arquivos de telemetria processados: `6`
  - eventos analisados: `37164054`
  - match completo TIPO+EVENTO+SITUACAO+NIVEL: `0`
  - match por EVENTO: `105976`
  - fallback Is_Dont_Go=1: `19962`
  - eventos criticos finais: `107002`
  - tags com sobreposicao apontamento x evento: `34`
  - registros rotulados: `377907`
  - positivos `target_4h`: `70811` (`18.737679%`)

## Estrutura de operacao diaria

1. Atualizar os arquivos mensais de telemetria em `data/raw/datasets/datasets/telemetria/`.
2. Executar `make label`.
3. Validar `data/processed/labeled/labeling_report.json`:
   - `rows_total_events`
   - `rows_critical_events_final`
   - `target_4h_positive_rate_pct`
   - `tag_overlap_count`
4. Se houver queda abrupta no `target_4h_positive_rate_pct`, bloquear treino e abrir incidente de dados.

Status: `CONCLUIDA`
