# Etapa 4 - EDA orientada a decisao

Data: 02/05/2026

## Objetivo

Executar EDA reprodutivel para entender desbalanceamento, qualidade e comportamento operacional antes da engenharia de atributos.

## Entregaveis implementados

1. Pipeline de EDA em `src/eda/run_eda.py`.
2. Atualizacao do target `make eda` para execucao real.
3. Geracao automatica de:
   - `reports/eda_report.md`
   - figuras em `reports/figures/`
4. Testes unitarios da etapa em `tests/test_eda.py`.

## Validacao planejada

1. `make lint`
2. `make test`
3. `make eda`

## Resultado da validacao

- `make lint`: OK
- `make test`: OK
- `make eda`: OK
- Testes: `23 passed`
- Cobertura total em `src`: `82%`

## Artefatos gerados

1. `reports/eda_report.md`
2. `reports/figures/eda_target_distribution.png`
3. `reports/figures/eda_ciclos_por_hora.png`
4. `reports/figures/eda_top_frotas.png`
5. `reports/figures/eda_duracao_ciclo_hist.png`
6. `reports/figures/eda_top_classes.png`
7. `notebooks/01_eda.ipynb`

## Achados principais

1. Dataset rotulado analisado: `377907` registros.
2. Target `target_4h` com `70811` positivos (`18.737679%`) apos a correcao da fonte de rotulacao (3b).
3. Cobertura temporal de `2025-01-01` até `2025-07-01` (UTC).
4. Duração média de ciclo: `29.6854 min` (P95 `60.0 min`).

Status: `CONCLUIDA`
