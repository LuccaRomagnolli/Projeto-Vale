# Etapa 5 - Feature engineering sem leakage

Data: 02/05/2026

## Objetivo

Construir um dataset de features reutilizavel para modelagem a partir do target corrigido da Etapa 3b.

## Entregaveis implementados

1. Pipeline de features em `src/features/build_features.py`.
2. Features temporais:
   - `hora_do_dia`
   - `dia_da_semana`
   - `mes`
   - `turno`
   - `duracao_ciclo_min`
   - `is_fim_de_semana`
3. Rolling windows por Tag em 4h, 8h e 24h:
   - `n_alertas_Xh`
   - `duracao_media_ciclo_Xh`
   - `n_ciclos_Xh`
   - `freq_classe_atividade_Xh`
   - `dias_desde_ultimo_alerta`
4. Derivadas de eventos:
   - `n_precondicoes_satisfeitas_4h`
   - `nivel_maximo_evento_recente`
5. Encodings categoricos:
   - `Tag_freq`
   - `Operador_freq`
   - `Classe_target_enc`
   - one-hot para `Frota` e `Tipo`
6. Persistencia de saidas:
   - `data/processed/features/features_dataset.parquet`
   - `data/processed/features/feature_report.json`

## Garantia anti-leakage

As features historicas usam apenas eventos anteriores ao `Fim` do ciclo. As janelas rolling sao calculadas com `closed="left"`, evitando uso do proprio instante ou eventos futuros.

O `Classe_target_enc` tambem foi implementado como media historica cumulativa por classe, ordenada por tempo, para evitar uso de informacao futura.

## Validacao planejada

1. `make lint`
2. `make test`
3. `make features`

## Resultado da validacao

- `make lint`: OK
- `make test`: OK
- `make features`: OK
- Testes: `25 passed`
- Cobertura total em `src`: `82%`

## Resultado da execucao real

- Linhas: `377907`
- Colunas: `41`
- Positivos `target_4h`: `70811`
- Taxa de positivos: `18.737679%`
- Saida principal: `data/processed/features/features_dataset.parquet`
- Relatorio: `data/processed/features/feature_report.json`

Status: `CONCLUIDA`
