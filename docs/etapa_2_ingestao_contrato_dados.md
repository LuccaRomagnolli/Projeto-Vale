

# Mining Fleet Alert Anticipation

Antecipacao de alertas criticos **"Don't Go"** em equipamentos de mineracao,  
com foco em priorizacao operacional por `Tag`.



---

# Etapa 2 - Ingestao e contrato de dados

Data: 04/05/2026

## Objetivo

Garantir leitura confiavel dos apontamentos operacionais, com contrato minimo
de schema, padronizacao temporal e relatorio de qualidade antes da rotulacao.

## Entregaveis


| Entrega                 | Caminho                                                   |
| ----------------------- | --------------------------------------------------------- |
| Leitor contratual       | `src/data_loader.py`                                      |
| Orquestracao de dataset | `src/data/make_dataset.py`                                |
| Snapshot validado       | `data/processed/labeled/apontamentos_validado.parquet`    |
| Relatorio JSON          | `data/processed/labeled/quality_report_ingestao.json`     |
| Relatorio Markdown      | `data/processed/labeled/quality_report_ingestao.md`       |
| Testes                  | `tests/test_data_loader.py`, `tests/test_make_dataset.py` |


## Contrato de entrada


| Coluna obrigatoria | Uso                          |
| ------------------ | ---------------------------- |
| `Tag`              | Identificacao do equipamento |
| `Frota`            | Segmentacao operacional      |
| `Tipo`             | Categoria de equipamento     |
| `Inicio`           | Inicio do ciclo              |
| `Fim`              | Fim do ciclo                 |


## Resultado da execucao real


| Indicador                               | Valor             |
| --------------------------------------- | ----------------- |
| Registros processados                   | `377907`          |
| Nulos em colunas obrigatorias           | `0`               |
| Linhas duplicadas completas             | `0`               |
| Duplicatas por `Tag` + `Inicio` + `Fim` | `101` (`0.0267%`) |
| Datas `Inicio` invalidas                | `0`               |
| Datas `Fim` invalidas                   | `0`               |
| Duracoes negativas                      | `0`               |
| Duracoes acima de 24h                   | `0`               |


## Evidencias

- `data/processed/labeled/quality_report_ingestao.json`
- `data/processed/labeled/apontamentos_validado.parquet`
- `make label`

## Decisao

Status: `CONCLUIDA`. O dataset de apontamentos atende ao contrato minimo e esta
apto para rotulacao com eventos criticos.

---

Vale · Mining Operations · Fleet Alert Anticipation