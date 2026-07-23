# Análise Comparativa de Algoritmos de Clustering: NBA Player Archetypes

## Sobre

Este projeto compara três algoritmos de clustering (**K-Means**, **Gaussian Mixture Models (GMM)** e **Clustering Hierárquico Aglomerativo**) aplicados a dados de desempenho de jogadores da NBA, com o objetivo de identificar arquétipos de jogadores e determinar qual algoritmo produz agrupamentos de maior qualidade. A comparação é formal: métricas internas, estabilidade por bootstrap e teste estatístico de Friedman com pós-teste de Wilcoxon.

## Resultados principais

- **K-Means (k=3) vence em todas as métricas**: Silhouette 0,303, Davies-Bouldin 1,234, Calinski-Harabasz 132,06 e ARI bootstrap 0,84.
- **Diferença estatisticamente significativa**: Friedman χ² = 181,46 (p = 3,95e-40); os três pares de algoritmos diferem sob Wilcoxon com correção de Bonferroni.
- **Conclusão robusta**: mantém-se com 9 features em vez de 11 (ARI 0,84 entre variantes) e replica na temporada 2025-26 (p = 5,85e-37).
- **Três arquétipos identificados**: Big Men, Role Players e estrelas/organizadores, com baixa concordância com as posições oficiais (ARI 0,144).

## Dados

- **Fonte:** NBA Stats API (`nba_api`)
- **Temporada principal:** 2024-25 (370 jogadores) | **Replicação:** 2025-26 (379 jogadores)
- **Filtro:** jogadores com pelo menos 500 minutos na temporada regular
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
├── config.py                  # Configurações (features, parâmetros, paths)
├── main.py                    # Pipeline principal (10 etapas)
├── friedman_analysis.py       # Bootstrap pareado + Friedman + Wilcoxon
├── sensitivity_features.py    # Análise de sensibilidade (11 vs 9 features)
├── persistence_analysis.py    # Persistência dos arquétipos entre temporadas
├── requirements.txt           # Dependências com versões fixadas
│
├── src/
│   ├── data_collection.py     # Coleta via NBA API
│   ├── preprocessing.py       # Seleção de features + StandardScaler
│   ├── clustering.py          # K-Means, GMM, Hierárquico
│   ├── validation.py          # Métricas, estabilidade, comparação
│   └── visualization.py       # Elbow, BIC, correlação, silhouette, PCA, dendrograma
│
├── data/
│   ├── nba_players_2024_25.csv
│   └── nba_players_2025_26.csv
│
└── notebooks/
    └── analysis.ipynb         # Análise interativa
```

## Instalação

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Ambiente verificado: Python 3.14, com as versões fixadas do `requirements.txt`.

## Execução

### Pipeline completo

```bash
python main.py
```

Gera resultados em `results/` e visualizações em `figures/`.

### Análises estatísticas

```bash
python friedman_analysis.py      # teste de Friedman + Wilcoxon
python sensitivity_features.py   # sensibilidade 11 vs 9 features
python persistence_analysis.py   # persistência dos arquétipos 2024-25 -> 2025-26
```

### Notebook interativo

```bash
jupyter notebook notebooks/analysis.ipynb
```

O notebook reproduz todas as etapas do pipeline de forma interativa, com visualizações inline.

### Re-coletar dados da NBA API

```bash
python -c "from src.data_collection import collect_all_data; collect_all_data(season='2024-25')"
```

## Reprodutibilidade

Todos os pontos estocásticos usam `random_state=42`. Com as versões fixadas do `requirements.txt`, os números reportados acima são reproduzidos exatamente a cada execução.
