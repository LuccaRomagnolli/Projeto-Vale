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
  <img src="https://img.shields.io/badge/Modelo-Selecao-EF9F27?style=flat-square"/>
  <img src="https://img.shields.io/badge/Janela-4h-085041?style=flat-square"/>
  <img src="https://img.shields.io/badge/Split-70/15/15-888780?style=flat-square"/>
</p>

---

# Etapa 2 - Ingestao e contrato de dados

Data: 04/05/2026

## Objetivo

Garantir leitura confiavel dos apontamentos operacionais, com contrato minimo
de schema, padronizacao temporal e relatorio de qualidade antes da rotulacao.

## Entregaveis

| Entrega | Caminho |
|---|---|
| Leitor contratual | `src/data_loader.py` |
| Orquestracao de dataset | `src/data/make_dataset.py` |
| Snapshot validado | `data/processed/labeled/apontamentos_validado.parquet` |
| Relatorio JSON | `data/processed/labeled/quality_report_ingestao.json` |
| Relatorio Markdown | `data/processed/labeled/quality_report_ingestao.md` |
| Testes | `tests/test_data_loader.py`, `tests/test_make_dataset.py` |

## Contrato de entrada

| Coluna obrigatoria | Uso |
|---|---|
| `Tag` | Identificacao do equipamento |
| `Frota` | Segmentacao operacional |
| `Tipo` | Categoria de equipamento |
| `Inicio` | Inicio do ciclo |
| `Fim` | Fim do ciclo |

## Resultado da execucao real

| Indicador | Valor |
|---|---:|
| Registros processados | `377907` |
| Nulos em colunas obrigatorias | `0` |
| Linhas duplicadas completas | `0` |
| Duplicatas por `Tag` + `Inicio` + `Fim` | `101` (`0.0267%`) |
| Datas `Inicio` invalidas | `0` |
| Datas `Fim` invalidas | `0` |
| Duracoes negativas | `0` |
| Duracoes acima de 24h | `0` |

## Evidencias

- `data/processed/labeled/quality_report_ingestao.json`
- `data/processed/labeled/apontamentos_validado.parquet`
- `make label`

## Decisao

Status: `CONCLUIDA`. O dataset de apontamentos atende ao contrato minimo e esta
apto para rotulacao com eventos criticos.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
