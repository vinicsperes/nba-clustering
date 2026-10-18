# Análise Comparativa de Algoritmos de Clustering: NBA Player Archetypes

## Sobre

Este projeto compara três algoritmos de clustering (**K-Means**, **Gaussian Mixture Models (GMM)** e **Clustering Hierárquico Aglomerativo**) aplicados a dados de desempenho de jogadores da NBA, com o objetivo de identificar arquétipos de jogadores e determinar qual algoritmo produz agrupamentos de maior qualidade. A comparação é formal: métricas internas, estabilidade por bootstrap e teste estatístico de Friedman com pós-teste de Wilcoxon.

## Resultados principais (temporada 2024-25)

- **K-Means (k=3) vence em todas as métricas**: Silhouette 0,301, Davies-Bouldin 1,261, Calinski-Harabasz 147,11 e ARI bootstrap 0,847.
- **Diferença estatisticamente significativa**: Friedman χ² = 122,46 (p = 2,56e-27); os três pares de algoritmos diferem sob Wilcoxon com correção de Bonferroni.
- **Conclusão robusta**: mantém-se com 9 features em vez de 11, com o recorte alternativo de 15 minutos por jogo e na replicação em 2025-26 (p = 2,86e-31).
- **Três arquétipos**: jogadores de função, estrelas/organizadores e jogadores de garrafão, com baixa concordância com as posições oficiais (ARI 0,132). 81,7% dos jogadores presentes nas duas temporadas mantêm o arquétipo (39,3% por acaso).

## Dados

- **Fonte:** NBA Stats API (`nba_api`), endpoint `leaguedashplayerstats`, temporada regular
- **Temporada principal:** 2024-25 (375 jogadores) | **Replicação:** 2025-26 (378 jogadores)
- **Filtro:** pelo menos 500 minutos jogados na temporada (minutos totais da API, sem arredondamento)
- **Médias por jogo:** totais da temporada divididos pelo número de jogos
- **Posições:** congeladas em `data/posicoes.csv` (a NBA não publica a posição por temporada)
- **11 features de box-score:**

| Categoria | Features |
|-----------|----------|
| Pontuação | PTS |
| Rebotes | REB, OREB, DREB |
| Distribuição | AST |
| Defesa | STL, BLK |
| Eficiência | FG_PCT, FG3_PCT, FT_PCT |
| Erros | TOV |

## Estrutura

```
├── config.py                  # Temporadas, filtro, features, parâmetros
├── main.py                    # Pipeline completo (gera tudo em results/)
├── friedman_analysis.py       # Bootstrap pareado + Friedman + Wilcoxon
├── sensitivity_features.py    # Sensibilidade às features (11 vs 9)
├── persistence_analysis.py    # Persistência dos arquétipos entre temporadas
├── requirements.txt           # Dependências com versões fixadas
│
├── src/
│   ├── data_collection.py     # Coleta via NBA API
│   ├── preprocessing.py       # Seleção de features + StandardScaler
│   ├── clustering.py          # K-Means, GMM, Hierárquico
│   ├── validation.py          # Métricas, estabilidade, testes estatísticos
│   ├── visualization.py       # As oito figuras da monografia
│   └── report.py              # relatorio.md a partir do numeros.json
│
└── data/
    ├── nba_players_2024_25.csv        # principal (500 min)
    ├── nba_players_2024_25_15mpg.csv  # sensibilidade ao recorte (15 min/jogo)
    ├── nba_players_2025_26.csv        # replicação
    └── posicoes.csv
```

## Instalação

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Ambiente verificado: Python 3.14, com as versões fixadas do `requirements.txt`.

## Execução

```bash
python main.py
```

Roda as três rodadas (2024-25, 2024-25 com 15 min/jogo e 2025-26) e a persistência. Cada rodada grava em `results/<rodada>/`:

| Arquivo | Conteúdo |
|---|---|
| `numeros.json` | todos os números citados na monografia |
| `relatorio.md` | os mesmos números em tabelas, para leitura |
| `atribuicoes.csv` | cluster de cada jogador em cada algoritmo |
| `bootstrap_pareado.csv` | os 100 ARIs pareados do teste de Friedman |
| `figuras/` | as oito figuras da monografia |

Uma rodada só: `python main.py --season 2025-26` (ou `--filtro 15mpg`). Os scripts de análise também rodam sozinhos, com `--season`.

### Re-coletar os dados

```bash
python src/data_collection.py --season 2024-25
python src/data_collection.py --season 2024-25 --filtro 15mpg
python src/data_collection.py --season 2025-26
```

## Reprodutibilidade

Todos os pontos estocásticos usam `random_state=42`. Com os CSVs de `data/` e as versões fixadas do `requirements.txt`, os números reportados acima são reproduzidos exatamente a cada execução.
