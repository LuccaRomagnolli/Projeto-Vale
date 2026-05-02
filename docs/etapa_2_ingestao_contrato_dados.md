# Etapa 2 - Ingestao e contrato de dados

Data: 30/04/2026

## Objetivo

Garantir leitura confiavel de apontamentos com contrato de schema, padronizacao temporal e relatorio de qualidade.

## Entregaveis implementados

1. Novo modulo `src/data_loader.py` com pipeline de ingestao contratual.
2. Validacao de colunas obrigatorias: `Tag`, `Frota`, `Tipo`, `Inicio`, `Fim`.
3. Padronizacao de `Inicio` e `Fim` para UTC.
4. Relatorio de qualidade com nulos, duplicatas e outliers de duracao.
5. Persistencia do snapshot validado em `data/processed/labeled/apontamentos_validado.parquet`.
6. Persistencia de relatorio em:
   - `data/processed/labeled/quality_report_ingestao.json`
   - `data/processed/labeled/quality_report_ingestao.md`
7. Orquestracao em `src/data/make_dataset.py`.
8. Testes unitarios da etapa em `tests/test_data_loader.py`.

## Validacao planejada

1. `make lint`
2. `make test`

## Resultado da validacao

- `make lint`: OK
- `make test`: OK
- `make label`: OK
- Testes: `13 passed`
- Cobertura total em `src`: `92%`
- Registros processados no run real: `377907`

Status: `CONCLUIDA`
