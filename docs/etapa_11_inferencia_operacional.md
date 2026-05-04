

# Mining Fleet Alert Anticipation

Antecipacao de alertas criticos **"Don't Go"** em equipamentos de mineracao,  
com foco em priorizacao operacional por `Tag`.



---

# Etapa 11 - Inferencia operacional

Data: 04/05/2026

## Objetivo

Documentar o contrato minimo para aplicar o artefato promovido em dados
operacionais e produzir scores, predicoes e ranking de risco.

## Entregaveis


| Entrega                | Caminho                                                |
| ---------------------- | ------------------------------------------------------ |
| Pipeline de inferencia | `src/inference.py`                                     |
| Artefato promovido     | `models/hist_gbdt_tuned.joblib`                        |
| Entrada padrao         | `data/processed/features/splits/features_test.parquet` |
| Saida padrao           | `reports/inference_scores.parquet`                     |
| Testes                 | `tests/test_inference.py`                              |


## Contrato do artefato

O arquivo `.joblib` deve conter:


| Chave             | Uso                                  |
| ----------------- | ------------------------------------ |
| `model`           | Estimador treinado                   |
| `feature_columns` | Ordem e nomes das features esperadas |
| `threshold`       | Threshold calibrado na validacao     |


## Contrato de entrada


| Formato | Condicao                     |
| ------- | ---------------------------- |
| CSV     | Arquivo tabular com features |
| Parquet | Arquivo tabular com features |


Durante a inferencia, o pipeline alinha o schema com `feature_columns`. Features
ausentes sao preenchidas com `0.0`, preservando a ordem esperada pelo modelo.

## Contrato de saida


| Coluna              | Descricao                                        |
| ------------------- | ------------------------------------------------ |
| `score`             | Probabilidade ou score de risco do modelo        |
| `prediction`        | Classe binaria apos aplicar threshold            |
| Colunas de contexto | Campos de entrada preservados quando disponiveis |


## Execucao

```bash
make infer
```

## Resultado esperado


| Indicador | Valor atual                        |
| --------- | ---------------------------------- |
| Modelo    | `hist_gbdt_tuned`                  |
| Threshold | `0.141388104973226`                |
| Features  | `48`                               |
| Saida     | `reports/inference_scores.parquet` |


## Decisao

Status: `CONCLUIDA`. A inferencia esta pronta para uso assistido, desde que o
ranking operacional priorize `TopK Tag-dia` e o monitoramento acompanhe drift,
volume de alertas e segmentos raros.

---

Vale · Mining Operations · Fleet Alert Anticipation