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

# Etapa 6 - Validacao temporal e baseline

Data: 02/05/2026

## Objetivo

Implementar split temporal 70/15/15 e baseline heuristico para servir como referencia minima antes dos modelos supervisionados.

## Entregaveis implementados

1. Split temporal em `src/models/validation.py`.
2. Persistencia dos splits:
   - `data/processed/features/splits/features_train.parquet`
   - `data/processed/features/splits/features_val.parquet`
   - `data/processed/features/splits/features_test.parquet`
3. Baseline heuristico em `src/models/train_baseline.py`.
4. Score do baseline:
   - `baseline_score = clip(n_alertas_24h / 24, 0, 1)`
5. Threshold calibrado na validacao temporal com alvo de recall minimo.
6. Artefatos:
   - `models/baseline_heuristico.joblib`
   - `reports/baseline_report.json`
   - `reports/baseline_scores.parquet`
7. Testes unitarios:
   - `tests/test_validation.py`
   - `tests/test_train_baseline.py`

## Validacao planejada

1. `make lint`
2. `make test`
3. `make train`

## Resultado da validacao

- `make lint`: OK
- `make test`: OK
- `make train`: OK
- Testes: `30 passed`
- Cobertura total em `src`: `84%`

## Resultado do split temporal

- Total: `377907`
- Treino: `264534` registros, `55060` positivos (`20.81396%`)
- Validacao: `56686` registros, `7261` positivos (`12.809159%`)
- Teste: `56687` registros, `8490` positivos (`14.976979%`)
- Treino: `2025-01-01 03:03:43+00:00` ate `2025-05-06 06:00:00+00:00`
- Validacao: `2025-05-06 06:00:00+00:00` ate `2025-06-02 15:00:00+00:00`
- Teste: `2025-06-02 15:00:00+00:00` ate `2025-07-01 03:00:00+00:00`

## Resultado do baseline

- Threshold calibrado na validacao: `0.0`
- Recall teste: `1.0`
- Precision teste: `0.14976978848766032`
- F1 teste: `0.2605213495558249`
- AUC-PR teste: `0.14976978848766032`
- AUC-ROC teste: `0.5`

## Leitura tecnica

O baseline atingiu recall maximo porque o threshold `0.0` classifica todos os registros como positivos. Isso e aceitavel como piso conservador, mas confirma que a heuristica `n_alertas_24h / 24` ainda nao discrimina bem o risco. O modelo principal da Etapa 7 deve superar principalmente AUC-PR e precision mantendo recall alto.

Status: `CONCLUIDA`
