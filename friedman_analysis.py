"""
Teste estatístico formal de comparação dos algoritmos: Friedman + Wilcoxon.

O teste de Friedman (Demsar, 2006, JMLR) compara os três algoritmos sobre
as mesmas medidas pareadas. Aqui, as medidas são os ARIs de 100 réplicas de
bootstrap do mesmo conjunto de dados; por isso o p-valor deve ser lido como
evidência de diferença sob reamostragem (ver a discussão dos limites na
monografia).

Procedimento:
  1. Determina o k de cada algoritmo pelo seu critério próprio.
  2. Roda 100 iterações de bootstrap *pareado*: em cada iteração, a mesma
     amostra é usada pelos três algoritmos.
  3. Calcula o ARI de cada partição contra a partição de referência
     (ajustada sobre o conjunto completo).
  4. Aplica o teste de Friedman e, se p < 0,05, o Wilcoxon pareado com
     correção de Bonferroni.

Uso:
    python friedman_analysis.py                    # temporada principal
    python friedman_analysis.py --season 2025-26   # replicação
"""

import argparse
import json

import numpy as np

from src.data_collection import load_data
from src.preprocessing import prepare_data
from src.clustering import run_clustering
from src.validation import (
    compute_paired_bootstrap_aris,
    friedman_test,
    wilcoxon_pairwise,
)
from config import SEASON, DEFAULT_FILTER, results_path


def run_friedman(X: np.ndarray, ks: dict):
    """Bootstrap pareado, teste de Friedman e pós-teste de Wilcoxon."""
    ari_df = compute_paired_bootstrap_aris(
        X,
        algorithms=["kmeans", "gmm", "hierarchical"],
        ks=ks,
        n_bootstrap=100,
        seed=42,
    )
    fried = friedman_test(ari_df)
    wilc = wilcoxon_pairwise(ari_df, alpha=0.05)
    return ari_df, fried, wilc


def main():
    parser = argparse.ArgumentParser(description="Teste de Friedman + Wilcoxon.")
    parser.add_argument("--season", default=SEASON)
    parser.add_argument("--filtro", default=DEFAULT_FILTER)
    args = parser.parse_args()

    df = load_data(args.season, args.filtro)
    X, _ = prepare_data(df)

    print("\n[1] Determinando k de cada algoritmo...")
    clustering = run_clustering(X)
    ks = {algo: clustering[algo].k for algo in ["kmeans", "gmm", "hierarchical"]}
    print(f"  k usados: {ks}")

    print("\n[2] Bootstrap pareado (100 iterações, mesma amostra para os 3)...")
    ari_df, fried, wilc = run_friedman(X, ks)
    print(ari_df.describe().round(4).to_string())

    print("\n[3] Teste de Friedman")
    print(f"  Estatística : {fried['statistic']:.4f}")
    print(f"  p-valor     : {fried['p_value']:.6e}")
    print(f"  Médias ARI  : {fried['means']}")
    print(f"  H0 rejeitada (alpha=0.05): {fried['reject_h0']}")

    print("\n[4] Wilcoxon pareado com Bonferroni (pós-teste)")
    print(wilc.round(6).to_string(index=False))

    out_dir = results_path(args.season, args.filtro)
    out_dir.mkdir(parents=True, exist_ok=True)
    ari_df.to_csv(out_dir / "bootstrap_pareado.csv", index=False)
    with open(out_dir / "friedman.json", "w") as f:
        json.dump({**fried, "ks": ks, "wilcoxon": wilc.to_dict("records")}, f, indent=2)
    print(f"\nResultados salvos em {out_dir}/")


if __name__ == "__main__":
    main()
