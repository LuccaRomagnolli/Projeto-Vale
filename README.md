# Projeto Vale - Análise Avançada de Dados

Estrutura básica do projeto criada com base no **Estudo Guiado - Análise Avançada de Dados** (desafio de antecipação de alertas críticos em frotas de mineração).

## Objetivo
Organizar o fluxo analítico ponta a ponta:
1. Entendimento do negócio
2. Entendimento dos dados (EDA)
3. Preparação dos dados
4. Modelagem
5. Avaliação
6. Resultados e conclusão

## Estrutura de pastas
```text
Projeto Vale/
├── Base de Dados/                # Arquivos originais recebidos
├── data/
│   ├── raw/                      # Cópias brutas para análise
│   ├── interim/                  # Dados intermediários
│   ├── processed/                # Dados prontos para modelagem
│   └── external/                 # Dados externos (se aplicável)
├── docs/
│   └── template_relatorio.md     # Template alinhado aos CM 1.x a 6.x
├── notebooks/
│   └── README.md                 # Sugestão de notebooks por etapa
├── reports/
│   └── figures/                  # Gráficos finais do relatório
├── src/
│   ├── data/
│   │   └── make_dataset.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_baseline.py
│   │   └── train_model.py
│   ├── evaluation/
│   │   └── evaluate_model.py
│   └── utils/
│       └── config.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Próximos passos sugeridos
1. Copiar os datasets da pasta `Base de Dados/` para `data/raw/` mantendo os originais intactos.
2. Preencher o notebook `01_business_understanding` com CM 1.1 e 1.2.
3. Implementar a rotulação inicial (`don't go`) em `src/data/make_dataset.py`.
4. Seguir o template em `docs/template_relatorio.md` para não perder nenhum conteúdo mínimo.
