<p align="center">
  <img src="../pictures/vale-logo-removebg-preview.png" alt="Vale" width="120"/>
</p>

<h1 align="center">Antecipação de Alertas Críticos em Frota de Mineração</h1>

<p align="center">
  Simulação de cenário operacional e econômico do piloto
</p>

---

# Simulação do piloto

Data: 29/08/2026

## Objetivo

Demonstrar como o sistema funcionaria numa operação real, incluindo as duas
peças que o conjunto de dados não possui: **registro de manutenção** e **dados
financeiros**.

Toda saída deste módulo é **simulação**. A separação entre o que foi medido e o
que é premissa está codificada na estrutura dos artefatos, e não apenas em
avisos de texto.

## O que é medido e o que é suposto

| Elo da cadeia | Origem |
|---|---|
| Casos críticos antecipados a mais que o acaso | **MEDIDO** — 220 em 30 dias, no conjunto de teste |
| Fração desses casos que a inspeção realmente evita | **PREMISSA** — nunca observada |
| Horas de parada por evento | **PREMISSA** |
| Custo por hora de parada | **PREMISSA** — faixa de literatura pública |
| Custo por inspeção | **PREMISSA** |

Apenas o primeiro elo é resultado. Todos os demais são suposições que a Vale
precisa fornecer antes de qualquer decisão de investimento.

### Por que a eficácia da prevenção não pode ser medida aqui

O conjunto não registra intervenções. Não existe, em lugar nenhum dos dados, a
observação "este equipamento foi inspecionado e o alerta Don't Go não ocorreu".
Sem esse contrafactual, é impossível separar **priorizar** de **prevenir**: o
modelo prova que ordena bem o risco, não que a ação muda o desfecho.

## Faixas de referência pública

Ordens de grandeza para caminhões fora-de-estrada em mineração. **Não são
números da operação da Vale.**

| Parâmetro | Faixa |
|---|---|
| Parada não planejada | US$ 5.000 a 20.000 por hora, dirigida pela produção perdida |
| Propagação pelo ciclo de lavra | Fontes citam até US$ 100.000/h quando a interrupção se espalha |
| Reparo emergencial | Cerca de `4,8x` o custo de um reparo planejado |

Fontes:
[Cummins](https://www.cummins.com/news/2021/03/23/reducing-machine-downtime-mining),
[HVI](https://heavyvehicleinspection.com/fleet-management/uptime/mining-uptime-executive-brief),
[MapTrack](https://www.maptrack.com/statistics/equipment-downtime-cost).

## O resultado mais defensável: ponto de equilíbrio

O simulador inverte a incógnita. Em vez de afirmar um retorno, responde **o
quanto a inspeção precisaria funcionar para o piloto empatar**:

| Custo por hora de parada | Eficácia mínima para empatar |
|---:|---:|
| US$ 5.000 | `2,7%` |
| US$ 12.500 | `1,1%` |
| US$ 20.000 | `0,7%` |

Este é o número a levar para a operação, porque a equipe de manutenção consegue
julgá-lo sem nenhum dado financeiro: *"das inspeções que fazemos a partir de um
alerta antecipado, evitamos a parada em mais de 1% dos casos?"*

Se a resposta for sim — e a experiência de campo sugere que seja folgadamente —
o piloto se paga. O retorno específico depende dos números da Vale.

Reproduzir: `make simulate`

## Manutenção simulada: o que ela demonstra e o que não demonstra

`src/simulation/maintenance.py` gera ordens de serviço sintéticas — preventivas
em intervalo regular e corretivas após uma fração dos dias com evento crítico.
O volume gerado é de `498` ordens, ou `1,8` por Tag por mês, faixa
operacionalmente plausível.

O modelo de demonstração foi treinado com duas features derivadas
(`dias_desde_ultima_manutencao` e `manutencao_corretiva_recente`), sobre o mesmo
split e os mesmos hiperparâmetros do artefato promovido:

| Braço | Features | `precision@15` | `recall@15` | `lift@15` |
|---|---:|---:|---:|---:|
| Sem manutenção (dados reais) | `44` | `0.8133` | `0.8862` | `2.5010` |
| Com manutenção (SIMULADA) | `46` | `0.8133` | `0.8862` | `2.5010` |

**O ganho foi exatamente zero, e isso é o resultado correto.**

As features estavam de fato na matriz — `46` colunas, todas as `56.689` linhas
de teste preenchidas — e o modelo lhes atribuiu importância de `0,001` e
`0,0003`, contra `0,30` da feature principal. Ele simplesmente as ignorou.

A razão é estrutural: o dado sintético foi gerado **independentemente do alvo**.
Não há nele nenhuma informação sobre alertas futuros, porque nenhuma foi
colocada. O modelo acertou ao descartá-lo.

> **Dado sintético não responde "essa feature ajudaria".** Ele só responde "o
> pipeline lida corretamente com essa feature". A primeira pergunta exige dado
> real de manutenção; nenhuma simulação pode substituí-la.
>
> Fabricar correlação com o alvo produziria um ganho — e esse ganho seria
> inteiramente artefato da geração, não evidência de nada.

### O que a demonstração entrega, então

1. O pipeline aceita e processa a feature de manutenção sem alteração estrutural
2. A causalidade é preservada: apenas ordens anteriores ao ciclo informam a feature
3. Ciclo sem histórico recebe `NaN`, não zero — zero significaria "manutenção hoje"
4. Quando o dado real chegar, o caminho já está pronto e testado

## Separação em relação aos dados reais

| Proteção | Implementação |
|---|---|
| Localização | `data/simulado/`, nunca em `data/processed/` |
| Nome do arquivo | Prefixo `SIMULADO_` |
| Conteúdo | Coluna `origem_dado = "SIMULADO"` em toda linha |
| Modelo | `models/demo/DEMO_*`, fora do gate de promoção |
| Relatórios | Campo `_aviso` e sufixo `_simulado` nos valores monetários |
| Testes | Verificam a marcação, a plausibilidade do volume e a causalidade |

**O artefato promovido em `models/model_selected.joblib` continua treinado
exclusivamente com dados reais.** As métricas oficiais do README não foram
afetadas por nada deste documento.

## Próximo passo para medir de verdade

O piloto assistido já registra as tratativas no painel — o que foi inspecionado,
por quem, e o achado. Depois de alguns meses de operação, esse registro fornece
exatamente o contrafactual que falta hoje:

> equipamentos apontados **e** inspecionados → o alerta ocorreu?
> equipamentos apontados **e não** inspecionados → o alerta ocorreu?

Com essa comparação, a eficácia da prevenção deixa de ser premissa e vira
medição. É o caminho para transformar este documento de simulação em evidência.

---

<p align="center">
  <sub>Vale · Mining Operations · Fleet Alert Anticipation</sub>
</p>
