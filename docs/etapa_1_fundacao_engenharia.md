# Etapa 1 - Fundacao de engenharia e qualidade

Data: 30/04/2026

## Objetivo

Criar base tecnica para desenvolvimento robusto e reprodutivel.

## Entregaveis implementados

1. `pyproject.toml` com configuracao de `black`, `ruff` e `pytest`.
2. `.env.example` com variaveis de ambiente do projeto.
3. `Makefile` com targets de qualidade e pipeline.
4. CI via GitHub Actions em `.github/workflows/ci.yml`.
5. Suite inicial de testes unitarios em `tests/`.
6. `requirements.txt` com versoes fixadas.
7. `.gitignore` para reduzir ruido no repositorio.

## Validacao planejada

1. `make format`
2. `make lint`
3. `make test`

## Resultado da validacao

- `make format`: OK
- `make lint`: OK
- `make test`: OK
- Testes: `9 passed`
- Cobertura total em `src`: `88%`

Status: `CONCLUIDA`
