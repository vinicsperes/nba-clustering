"""
Análise de sensibilidade à escolha das features (multicolinearidade).

Roda os três algoritmos com dois conjuntos de features:
  (a) 11 originais: PTS, REB, OREB, DREB, AST, STL, BLK, FG_PCT, FG3_PCT, FT_PCT, TOV
  (b)  9 sem redundância: tira OREB e DREB (mantém só REB)

Por quê? REB = OREB + DREB. Manter as três triplica o peso da dimensão
"rebote" no cálculo de distância euclidiana. Para cada algoritmo, compara o
k escolhido, o Silhouette e o ARI entre as partições das duas variantes.

Uso:
    python sensitivity_features.py [--season 2024-25]
"""

import argparse

from sklearn.metrics import adjusted_rand_score

from src.data_collection import load_data
from src.preprocessing import prepare_data
from src.clustering import run_clustering, GMMClusterer
from config import SEASON, DEFAULT_FILTER, CLUSTERING_FEATURES

FEATURES_11 = CLUSTERING_FEATURES
FEATURES_9 = [f for f in CLUSTERING_FEATURES if f not in {"OREB", "DREB"}]


def _fit_variant(df, features):
    X, _ = prepare_data(df, features=features)
    clustering = run_clustering(X)
    return {
        "kmeans": clustering["kmeans"],
        "gmm": clustering["gmm"],
        "gmm_silhouette": GMMClusterer(selection_metric="silhouette").fit(X).get_best_result(),
        "hierarchical": clustering["hierarchical"],
    }


def run_sensitivity(df) -> dict:
    """k, Silhouette e ARI (11 vs 9) de cada algoritmo."""
    res_11 = _fit_variant(df, FEATURES_11)
    res_9 = _fit_variant(df, FEATURES_9)
    return {
        algo: {
            "k_11": int(res_11[algo].k),
            "silhouette_11": float(res_11[algo].silhouette),
            "k_9": int(res_9[algo].k),
            "silhouette_9": float(res_9[algo].silhouette),
            "ari_11_vs_9": float(adjusted_rand_score(res_11[algo].labels, res_9[algo].labels)),
        }
        for algo in res_11
    }


def main():
    parser = argparse.ArgumentParser(description="Sensibilidade às features.")
    parser.add_argument("--season", default=SEASON)
    parser.add_argument("--filtro", default=DEFAULT_FILTER)
    args = parser.parse_args()

    result = run_sensitivity(load_data(args.season, args.filtro))
    print("\nalgoritmo        k(11)  Sil(11)  k(9)  Sil(9)  ARI(11 vs 9)")
    for algo, r in result.items():
        print(f"{algo:15s}  {r['k_11']:5d}  {r['silhouette_11']:7.3f}  {r['k_9']:4d}  "
              f"{r['silhouette_9']:6.3f}  {r['ari_11_vs_9']:12.3f}")


if __name__ == "__main__":
    main()
