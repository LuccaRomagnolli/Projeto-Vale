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

# Notebook principal e contexto do projeto

Data de atualizacao: 04/05/2026

## Objetivo

Documentar o papel do notebook principal como walkthrough tecnico-executivo do
projeto, sem manter referencias antigas a notebooks numerados que nao existem
na estrutura atual versionada.

## Notebook oficial

| Item | Caminho |
|---|---|
| Notebook principal | `notebooks/main.ipynb` |
| Dashboard auxiliar | `notebooks/qc_dashboard.png` |

## Fluxo consolidado no notebook

1. Contexto de negocio e pergunta operacional.
2. Ingestao e qualidade dos dados.
3. Rotulacao `target_4h`.
4. EDA e leitura de viabilidade.
5. Engenharia de features.
6. Benchmark e tuning de modelos.
7. Avaliacao `TopK Tag-dia`.
8. Analise segmentada e recomendacao final.

## Como reproduzir

```bash
make run-all
make smoke
```

Depois, abrir `notebooks/main.ipynb` e executar as celulas na ordem.

## Fontes usadas pelo notebook

- `README.md`
- `docs/politica_promocao_modelo.md`
- `docs/controle_alteracoes.md`
- `reports/model_benchmark_report.json`
- `reports/hist_gbdt_tuning_report.json`
- `reports/operational_metrics_report.json`
- `reports/segment_operational_report.json`

## Decisao

Status: `VIGENTE`. O projeto usa um notebook principal consolidado. Referencias
a notebooks `01..09` foram removidas por nao refletirem a estrutura atual.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
