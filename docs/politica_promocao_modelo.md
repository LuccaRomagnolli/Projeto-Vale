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

# Politica unica de promocao de modelo

Data: 04/05/2026

## Objetivo

Padronizar a decisao de promocao para evitar ambiguidade entre baseline,
candidatos oficiais e artefato operacional vigente.

## Regra obrigatoria

Um modelo so pode ser promovido quando todas as condicoes abaixo forem verdadeiras:

| Condicao | Regra |
|---|---|
| Selecao | Modelo selecionado por validacao temporal operacional entre 3 candidatos oficiais |
| Metrica primaria | Maior `val_top15_recall_at_k` |
| Desempates | `val_top15_precision_at_k`, `val_top15_lift_vs_random`, `val_auc_pr` |
| Precision minima no teste | `Precision@Top15 Tag-dia >= 0.60` |
| Recall minimo no teste | `Recall@Top15 Tag-dia >= 0.70` |
| Lift minimo no teste | `Lift@Top15 Tag-dia >= 1.90` |
| Estabilidade | `make gate-stability` aprovado |
| Segmentos raros | Marcados como inconclusivos, nao como falha global automatica |

## Gate de estabilidade

| Metrica | Limite |
|---|---:|
| `std(test_top15_recall_at_k)` | `<= 0.03` |
| `std(test_top15_precision_at_k)` | `<= 0.05` |

## Fluxo oficial

```bash
make model-selection
make gate-stability
make evaluate
make evaluate-segments
make infer
```

## Artefato operacional vigente

| Campo | Valor |
|---|---|
| Modelo | `hist_gbdt_optuna`, definido em `reports/model_selection/model_selection_report.json` |
| Artefato | `models/model_selected.joblib` |
| Threshold | calibrado na validacao e persistido no artefato |
| Test Precision@15 | `0.8133` |
| Test Recall@15 | `0.8862` |
| Test Lift@15 | `2.5010` |

> Valores atualizados em 29/08/2026 apos a correcao de vazamento temporal e o
> reparo das features de historico de alertas. Ver `docs/controle_alteracoes.md`,
> alteracao 04, para a decomposicao medida do impacto.

## Evidencias minimas

- `reports/model_selection/model_selection_report.json`
- `reports/model_selection/model_selection_trials.csv`
- `reports/model_selection/model_selection_backtest_report.csv`
- `reports/model_selection/model_selected_threshold_curve.csv`
- `reports/operational/operational_metrics_report.json`
- `reports/segments/segment_operational_report.json`
- `reports/model_selection/leakage_ablation_report.json`

## Decisao

Status: `VIGENTE`. O projeto esta metodologicamente apto para piloto operacional
assistido, com monitoramento continuo de drift, volume de alertas e segmentos
raros.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
