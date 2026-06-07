"""
Análise de sensibilidade — multicolinearidade.

Roda o pipeline com dois conjuntos de features:
  (a) 11 originais: PTS, REB, OREB, DREB, AST, STL, BLK, FG_PCT, FG3_PCT, FT_PCT, TOV
  (b)  9 sem redundância: tira OREB e DREB (mantém só REB)

Por quê? REB = OREB + DREB. Manter as três triplica o peso da dimensão "rebote"
no cálculo de distância euclidiana, o que pode inflar a importância do cluster
Big Men. Este script verifica se a conclusão (K-Means vence em todas as métricas)
se mantém com 9 features.

Saída: tabela comparativa + ARI entre as duas particionamentos do K-Means.
"""

from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from src.data_collection import load_data
from src.preprocessing import prepare_data, get_position_labels
from src.clustering import KMeansClusterer, GMMClusterer, HierarchicalClusterer
from config import SEASON, ALL_FEATURES


FEATURES_11 = ALL_FEATURES
FEATURES_9 = [f for f in ALL_FEATURES if f not in {"OREB", "DREB"}]


def run_variant(df, features, label):
    print(f"\n{'#' * 70}")
    print(f"# VARIANTE: {label}  ({len(features)} features)")
    print(f"# {features}")
    print(f"{'#' * 70}")

    X, df_meta, preprocessor = prepare_data(df, features=features)

    # K-Means (Silhouette)
    km = KMeansClusterer().fit(X)
    km_r = km.get_best_result()

    # GMM por BIC (mudança da Tarefa 2)
    gmm = GMMClusterer(selection_metric="bic").fit(X)
    gmm_r = gmm.get_best_result()

    # GMM por Silhouette também (pra comparação fair com a tabela antiga)
    gmm_sil = GMMClusterer(selection_metric="silhouette").fit(X)
    gmm_sil_r = gmm_sil.get_best_result()

    # Hierárquico (Silhouette)
    hier = HierarchicalClusterer().fit(X)
    hier_r = hier.get_best_result()

    def metrics(r):
        return {
            "k": r.k,
            "Sil": silhouette_score(X, r.labels),
            "DB": davies_bouldin_score(X, r.labels),
            "CH": calinski_harabasz_score(X, r.labels),
        }

    print(f"\n=== Métricas — {label} ===")
    print(f"K-Means         k={km_r.k}: {metrics(km_r)}")
    print(f"GMM (BIC)       k={gmm_r.k}: {metrics(gmm_r)}")
    print(f"GMM (Sil)       k={gmm_sil_r.k}: {metrics(gmm_sil_r)}")
    print(f"Hierárquico     k={hier_r.k}: {metrics(hier_r)}")

    return {
        "X": X,
        "kmeans": km_r,
        "gmm_bic": gmm_r,
        "gmm_sil": gmm_sil_r,
        "hier": hier_r,
    }


def main():
    df = load_data(season=SEASON)
    positions = get_position_labels(df)

    res_11 = run_variant(df, FEATURES_11, "11 features (original)")
    res_9 = run_variant(df, FEATURES_9, "9 features (sem OREB/DREB)")

    print("\n" + "=" * 70)
    print("COMPARAÇÃO ENTRE AS DUAS VARIANTES (mesmo algoritmo)")
    print("=" * 70)
    print("ARI compara o particionamento do K-Means(k=3) entre 11 e 9 features.")
    print("ARI próximo de 1 = mesmas atribuições; próximo de 0 = aleatório.\n")

    ari_km = adjusted_rand_score(res_11["kmeans"].labels, res_9["kmeans"].labels)
    ari_hier = adjusted_rand_score(res_11["hier"].labels, res_9["hier"].labels)
    ari_gmm_sil = adjusted_rand_score(res_11["gmm_sil"].labels, res_9["gmm_sil"].labels)

    print(f"  K-Means       ARI(11 vs 9) = {ari_km:.4f}")
    print(f"  Hierárquico   ARI(11 vs 9) = {ari_hier:.4f}")
    print(f"  GMM (Sil)     ARI(11 vs 9) = {ari_gmm_sil:.4f}")

    print("\n" + "=" * 70)
    print("DELTA SILHOUETTE (9 menos 11)")
    print("=" * 70)
    s11_km = silhouette_score(res_11["X"], res_11["kmeans"].labels)
    s9_km = silhouette_score(res_9["X"], res_9["kmeans"].labels)
    s11_h = silhouette_score(res_11["X"], res_11["hier"].labels)
    s9_h = silhouette_score(res_9["X"], res_9["hier"].labels)
    print(f"  K-Means        Sil 11={s11_km:.3f}  Sil 9={s9_km:.3f}  Δ={s9_km - s11_km:+.3f}")
    print(f"  Hierárquico    Sil 11={s11_h:.3f}  Sil 9={s9_h:.3f}  Δ={s9_h - s11_h:+.3f}")

    # Comparação com posições — concordância continua igual?
    ari_km_pos_11 = adjusted_rand_score(positions[positions != "Unknown"],
                                         res_11["kmeans"].labels[positions != "Unknown"])
    ari_km_pos_9 = adjusted_rand_score(positions[positions != "Unknown"],
                                        res_9["kmeans"].labels[positions != "Unknown"])
    print(f"\n  K-Means ARI vs posições  11={ari_km_pos_11:.3f}  9={ari_km_pos_9:.3f}")


if __name__ == "__main__":
    main()
