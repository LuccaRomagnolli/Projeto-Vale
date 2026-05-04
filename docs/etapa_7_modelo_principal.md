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

# Etapa 7 - Modelo supervisionado principal

Data: 04/05/2026

## Objetivo

Treinar um modelo supervisionado para estimar risco de alerta critico em ate 4
horas, usando split temporal, threshold calibrado somente na validacao e
remocao explicita de colunas de vazamento.

## Entregaveis

| Entrega | Caminho |
|---|---|
| Treino principal | `src/models/train_model.py` |
| Artefato | `models/model_principal.joblib` |
| Relatorio | `reports/model_principal_report.json` |
| Scores | `reports/model_principal_scores.parquet` |
| Importancia de variaveis | `reports/model_feature_importance.csv` |
| Testes | `tests/test_train_model.py` |

## Modelo executado

| Campo | Valor |
|---|---|
| Nome | `modelo_principal_supervisionado` |
| Biblioteca | `lightgbm.LGBMClassifier` |
| Features usadas | `48` |
| Threshold calibrado | `0.10810492961676205` |
| TopK operacional primario | `15 Tags por dia` |

## Colunas removidas por vazamento ou identidade bruta

| Coluna | Motivo |
|---|---|
| `next_critical_event_time` | Informacao futura direta |
| `tte_horas` | Tempo ate evento futuro |
| `target_4h` | Variavel alvo |
| `Id`, `Inicio`, `Fim` | Identificadores e timestamps brutos |
| `Tag`, `Classe` | Substituidas por features historicas/encodings |

## Resultado operacional do modelo principal

| Split | Precision@15 | Recall@15 | Lift@15 | AUC-PR |
|---|---:|---:|---:|---:|
| Treino | `0.8688` | `0.8033` | `2.3233` | `0.7648` |
| Validacao | `0.6548` | `0.7412` | `2.1002` | `0.2392` |
| Teste | `0.6889` | `0.7506` | `2.1184` | `0.2810` |

## Decisao

Status: `CONCLUIDA`. O modelo principal supera o baseline, mas a promocao final
deve seguir benchmark, tuning HistGBDT, gate de estabilidade e avaliacao
operacional.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
