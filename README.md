# Análise Comparativa de Algoritmos de Clustering — NBA Player Archetypes

Código-fonte do Trabalho de Conclusão de Curso em Sistemas de Informação (IFC — Campus Araquari).

- **Autor:** Vinicius Peres
- **Orientador:** Paulo Cesar Fernandes de Oliveira

## Sobre

Este projeto compara três algoritmos de clustering — **K-Means**, **Gaussian Mixture Models (GMM)** e **Clustering Hierárquico Aglomerativo** — aplicados a dados de desempenho de jogadores da NBA (temporada 2024-25), com o objetivo de identificar arquétipos de jogadores e determinar qual algoritmo produz agrupamentos de maior qualidade.

## Dados

- **Fonte:** NBA Stats API (`nba_api`)
- **Temporada:** 2024-25
- **Filtro:** jogadores com ≥ 500 minutos totais
- **Dataset:** 370 jogadores, 11 features de box-score

| Categoria | Features |
|-----------|----------|
| Pontuação | PTS |
| Rebotes | REB, OREB, DREB |
| Distribuição | AST, TOV |
| Defesa | STL, BLK |
| Eficiência | FG_PCT, FG3_PCT, FT_PCT |

## Estrutura

```
├── config.py               # Configurações (features, parâmetros, paths)
├── main.py                 # Pipeline principal (10 etapas)
├── requirements.txt        # Dependências Python
│
├── src/
│   ├── data_collection.py  # Coleta via NBA API
│   ├── preprocessing.py    # Seleção de features + StandardScaler
│   ├── clustering.py       # K-Means, GMM, Hierárquico
│   ├── validation.py       # Métricas, estabilidade, comparação
│   └── visualization.py    # Gráficos (elbow, silhouette, PCA, dendrograma)
│
├── data/
│   └── nba_players_2024_25.csv
│
└── notebooks/
    └── analysis.ipynb      # Análise interativa
```

## Instalação

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Execução

### Pipeline completo

```bash
python main.py
```

Gera resultados em `results/` e visualizações em `figures/`.

### Notebook interativo

```bash
jupyter notebook notebooks/analysis.ipynb
```

O notebook reproduz todas as etapas do pipeline de forma interativa, com visualizações inline.

### Re-coletar dados da NBA API

```bash
python -c "from src.data_collection import collect_all_data; collect_all_data(season='2024-25')"
```

## Dependências

- Python 3.10+
- scikit-learn, numpy, pandas, matplotlib, seaborn, scipy, nba_api
- Jupyter (para o notebook)

Todas listadas em `requirements.txt`.
