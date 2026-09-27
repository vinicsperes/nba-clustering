"""
Teste estatístico formal de comparação dos algoritmos — Friedman + Wilcoxon.

O teste de Friedman (Demšar, 2006, JMLR) compara os três algoritmos sobre
as mesmas medidas pareadas. Aqui, as medidas são os ARIs de 100 réplicas de
bootstrap do mesmo conjunto de dados; por isso o p-valor deve ser lido como
evidência de diferença sob reamostragem (ver a discussão dos limites na
monografia).

Este script:
  1. Roda 100 iterações de bootstrap *pareado* — todas as iterações usam a
     mesma amostra para os três algoritmos.
  2. Calcula ARI vs labels de referência (treino completo) por iteração.
  3. Aplica teste de Friedman.
  4. Se p < 0.05, aplica Wilcoxon pareado com correção de Bonferroni.
"""

import json
from pathlib import Path

from src.data_collection import load_data
from src.preprocessing import prepare_data
from src.clustering import KMeansClusterer, GMMClusterer, HierarchicalClusterer
from src.validation import (
    compute_paired_bootstrap_aris,
    friedman_test,
    wilcoxon_pairwise,
)
from config import SEASON, RESULTS_DIR


def main():
    df = load_data(season=SEASON)
    X, df_meta, preprocessor = prepare_data(df)

    print("\n[1] Determinando k de cada algoritmo nas 11 features originais...")
    km = KMeansClusterer().fit(X).get_best_result()
    gmm_bic = GMMClusterer(selection_metric="bic").fit(X).get_best_result()
    gmm_sil = GMMClusterer(selection_metric="silhouette").fit(X).get_best_result()
    hier = HierarchicalClusterer().fit(X).get_best_result()

    ks_used = {
        "kmeans": km.k,
        "gmm": gmm_bic.k,
        "hierarchical": hier.k,
    }
    print(f"\n  K-Means       k={km.k}")
    print(f"  GMM (BIC)     k={gmm_bic.k}")
    print(f"  Hierárquico   k={hier.k}")

    print("\n[2] Bootstrap pareado (100 iterações, mesma amostra para os 3)...")
    ari_df = compute_paired_bootstrap_aris(
        X,
        algorithms=["kmeans", "gmm", "hierarchical"],
        ks=ks_used,
        n_bootstrap=100,
        seed=42,
    )

    print("\nResumo das 100 ARIs pareadas:")
    print(ari_df.describe().round(4).to_string())

    print("\n[3] Teste de Friedman")
    fried = friedman_test(ari_df)
    print(f"  Estatística : {fried['statistic']:.4f}")
    print(f"  p-valor     : {fried['p_value']:.6e}")
    print(f"  n blocos    : {fried['n_blocks']}")
    print(f"  Médias ARI  : {fried['means']}")
    print(f"  H0 rejeitada (alpha=0.05): {fried['reject_h0']}")

    print("\n[4] Wilcoxon pareado com Bonferroni (pós-teste)")
    wilc = wilcoxon_pairwise(ari_df, alpha=0.05)
    print(wilc.round(6).to_string(index=False))

    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    ari_df.to_csv(out_dir / "bootstrap_aris_paired.csv", index=False)
    wilc.to_csv(out_dir / "wilcoxon_pairwise.csv", index=False)
    with open(out_dir / "friedman_result.json", "w") as f:
        json.dump({**fried, "ks_used": ks_used,
                   "gmm_silhouette_k": gmm_sil.k}, f, indent=2)

    print(f"\nArquivos salvos em {out_dir}/")
    print("  - bootstrap_aris_paired.csv")
    print("  - wilcoxon_pairwise.csv")
    print("  - friedman_result.json")


if __name__ == "__main__":
    main()
