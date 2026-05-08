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

# Etapa 11 - Inferencia operacional

Data: 04/05/2026

## Objetivo

Documentar o contrato minimo para aplicar o artefato promovido em dados
operacionais e produzir scores, predicoes e ranking de risco.

## Entregaveis

| Entrega | Caminho |
|---|---|
| Pipeline de inferencia | `src/inference.py` |
| Artefato selecionado | `models/model_selected.joblib` |
| Entrada padrao | `data/processed/features/splits/features_test.parquet` |
| Saida padrao | `reports/inference_scores.parquet` |
| Testes | `tests/test_inference.py` |

## Contrato do artefato

O arquivo `.joblib` deve conter:

| Chave | Uso |
|---|---|
| `model` | Estimador treinado |
| `feature_columns` | Ordem e nomes das features esperadas |
| `threshold` | Threshold calibrado na validacao |
| `model_name` | Familia selecionada pela validacao operacional |
| `selection_rule` | Regra oficial de desempate e promocao |

## Contrato de entrada

| Formato | Condicao |
|---|---|
| CSV | Arquivo tabular com features |
| Parquet | Arquivo tabular com features |

Durante a inferencia, o pipeline alinha o schema com `feature_columns`. Features
ausentes sao preenchidas com `0.0`, preservando a ordem esperada pelo modelo.

## Contrato de saida

| Coluna | Descricao |
|---|---|
| `score` | Probabilidade ou score de risco do modelo |
| `prediction` | Classe binaria apos aplicar threshold |
| Colunas de contexto | Campos de entrada preservados quando disponiveis |

## Execucao

```bash
make infer
```

## Resultado esperado

| Indicador | Valor atual |
|---|---|
| Modelo | Lido de `models/model_selected.joblib` |
| Threshold | Lido de `models/model_selected.joblib` |
| Features | `48` |
| Saida | `reports/inference_scores.parquet` |

## Decisao

Status: `CONCLUIDA`. A inferencia esta pronta para uso assistido, desde que o
ranking operacional priorize `TopK Tag-dia` e o monitoramento acompanhe drift,
volume de alertas e segmentos raros.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
