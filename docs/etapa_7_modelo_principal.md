

# Mining Fleet Alert Anticipation

Antecipacao de alertas criticos **"Don't Go"** em equipamentos de mineracao,  
com foco em priorizacao operacional por `Tag`.



---

# Etapa 7 - Modelagem supervisionada e selecao tecnica

Data: 04/05/2026

## Objetivo

Executar modelagem supervisionada sem definir modelo principal a priori,
incluindo benchmark tecnico de 4 candidatos com multiplas iteracoes,
split temporal, threshold calibrado somente na validacao e remocao explicita
de colunas de vazamento.

## Entregaveis


| Entrega                           | Caminho                                  |
| --------------------------------- | ---------------------------------------- |
| Treino de referencia              | `src/models/train_model.py`              |
| Benchmark robusto                 | `src/models/benchmark_models.py`         |
| Artefato selecionado no benchmark | `models/model_benchmark_selected.joblib` |
| Relatorio consolidado             | `reports/model_benchmark_report.json`    |
| Scores benchmark                  | `reports/model_benchmark_scores.parquet` |
| Importancia de variaveis          | `reports/model_feature_importance.csv`   |
| Testes                            | `tests/test_train_model.py`              |


## Estrategia executada


| Campo                     | Valor                    |
| ------------------------- | ------------------------ |
| Politica                  | `sem principal a priori` |
| Candidatos benchmark      | `4 modelos`              |
| Iteracoes por candidato   | `3`                      |
| Features usadas           | `48`                     |
| TopK operacional primario | `15 Tags por dia`        |


## Colunas removidas por vazamento ou identidade bruta


| Coluna                     | Motivo                                         |
| -------------------------- | ---------------------------------------------- |
| `next_critical_event_time` | Informacao futura direta                       |
| `tte_horas`                | Tempo ate evento futuro                        |
| `target_4h`                | Variavel alvo                                  |
| `Id`, `Inicio`, `Fim`      | Identificadores e timestamps brutos            |
| `Tag`, `Classe`            | Substituidas por features historicas/encodings |


## Resultado operacional da selecao tecnica


| Split     | Precision@15 | Recall@15 | Lift@15  | AUC-PR   |
| --------- | ------------ | --------- | -------- | -------- |
| Treino    | `0.8688`     | `0.8033`  | `2.3233` | `0.7648` |
| Validacao | `0.6548`     | `0.7412`  | `2.1002` | `0.2392` |
| Teste     | `0.6889`     | `0.7506`  | `2.1184` | `0.2810` |


## Decisao

Status: `CONCLUIDA`. A selecao e feita apenas apos benchmark robusto (4 modelos,
multiplas iteracoes), seguida de tuning, gate de estabilidade e avaliacao
operacional.

---

Vale · Mining Operations · Fleet Alert Anticipation