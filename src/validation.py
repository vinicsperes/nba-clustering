"""
Módulo de validação dos clusters.

Métricas calculadas:
1. Internas: Silhouette Score, Calinski-Harabasz, Davies-Bouldin
2. Estabilidade: Multi-seed ARI (30 seeds), Bootstrap ARI (100 iterações)
3. Externa (secundária): ARI e NMI vs posições tradicionais (Guard/Forward/Center)
4. Comparação formal: bootstrap pareado, teste de Friedman e Wilcoxon pareado
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    adjusted_rand_score,
    normalized_mutual_info_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import VALIDATION_CONFIG, KMEANS_CONFIG, GMM_CONFIG, HIERARCHICAL_CONFIG


def compute_internal_metrics(
    X: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Calcula métricas de validação interna.

    Métricas:
    - Silhouette Score: Coesão + separação (-1 a 1, maior = melhor)
    - Calinski-Harabasz: Razão de dispersão (maior = melhor)
    - Davies-Bouldin: Similaridade intra/inter (menor = melhor)
    """
    sil = silhouette_score(X, labels)
    sil_samples = silhouette_samples(X, labels)

    sil_per_cluster = {
        int(c): float(sil_samples[labels == c].mean())
        for c in np.unique(labels)
    }

    return {
        "silhouette": float(sil),
        "silhouette_per_cluster": sil_per_cluster,
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


def compute_external_metrics(
    cluster_labels: np.ndarray,
    position_labels: np.ndarray
) -> Dict[str, float]:
    """
    Concordância entre clusters e posições tradicionais (análise secundária).

    ARI ~0 = independentes, ARI ~1 = idênticos. Jogadores sem posição
    conhecida ficam de fora.
    """
    known = position_labels != "Unknown"
    return {
        "ari": float(adjusted_rand_score(position_labels[known], cluster_labels[known])),
        "nmi": float(normalized_mutual_info_score(position_labels[known], cluster_labels[known])),
        "n_samples": int(known.sum()),
    }


def _fit_algorithm(algorithm: str, X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Ajusta um algoritmo e retorna labels."""
    if algorithm == "kmeans":
        model = KMeans(
            n_clusters=k,
            n_init=KMEANS_CONFIG["n_init"],
            max_iter=KMEANS_CONFIG["max_iter"],
            random_state=seed,
        )
        return model.fit_predict(X)
    elif algorithm == "gmm":
        model = GaussianMixture(
            n_components=k,
            covariance_type=GMM_CONFIG["covariance_type"],
            n_init=GMM_CONFIG["n_init"],
            max_iter=GMM_CONFIG["max_iter"],
            random_state=seed,
        )
        model.fit(X)
        return model.predict(X)
    elif algorithm == "hierarchical":
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=HIERARCHICAL_CONFIG["linkage"],
        )
        return model.fit_predict(X)
    else:
        raise ValueError(f"Algoritmo desconhecido: {algorithm}")


def compute_stability(
    X: np.ndarray,
    algorithm: str,
    k: int,
    n_seeds: int = VALIDATION_CONFIG["n_seeds"],
    bootstrap_frac: float = VALIDATION_CONFIG["bootstrap_sample_fraction"],
    n_bootstrap: int = VALIDATION_CONFIG["bootstrap_iterations"],
) -> Dict[str, float]:
    """
    Calcula a estabilidade de um algoritmo de clustering.

    Multi-seed: roda com n_seeds sementes diferentes e calcula o ARI médio
    entre todos os pares de partições. O Hierárquico é determinístico e
    recebe 1,0 sem rodar.
    Bootstrap: amostra bootstrap_frac dos dados (com reposição) n_bootstrap
    vezes e compara cada partição com a de referência pelo ARI.
    """
    rng = np.random.RandomState(42)
    n_samples = X.shape[0]

    # --- Multi-seed stability ---
    seed_mean_ari = 1.0
    if algorithm != "hierarchical":
        seeds = rng.randint(0, 100000, size=n_seeds)
        all_labels = [_fit_algorithm(algorithm, X, k, seed=int(s)) for s in seeds]
        aris = [
            adjusted_rand_score(all_labels[i], all_labels[j])
            for i in range(len(all_labels))
            for j in range(i + 1, len(all_labels))
        ]
        seed_mean_ari = float(np.mean(aris))

    # --- Bootstrap stability ---
    bootstrap_size = int(n_samples * bootstrap_frac)
    reference_labels = _fit_algorithm(algorithm, X, k, seed=42)

    bootstrap_aris = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=bootstrap_size, replace=True)
        boot_labels = _fit_algorithm(algorithm, X[indices], k, seed=42)
        bootstrap_aris.append(adjusted_rand_score(reference_labels[indices], boot_labels))

    return {
        "seed_mean_ari": seed_mean_ari,
        "bootstrap_mean_ari": float(np.mean(bootstrap_aris)),
    }


def compute_paired_bootstrap_aris(
    X: np.ndarray,
    algorithms: List[str],
    ks: Dict[str, int],
    n_bootstrap: int = VALIDATION_CONFIG["bootstrap_iterations"],
    bootstrap_frac: float = VALIDATION_CONFIG["bootstrap_sample_fraction"],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap pareado para comparação estatística entre algoritmos.

    A diferença para `compute_stability` é que aqui *todas* as iterações usam
    a mesma amostra para *todos* os algoritmos. Isso produz três séries de
    ARIs pareadas por índice de iteração, formato exigido pelo teste de
    Friedman e pelo Wilcoxon.

    Returns
    -------
    pd.DataFrame
        Colunas = algoritmos, n_bootstrap linhas. Linha i = ARI de cada
        algoritmo na iteração i (mesma amostra).
    """
    rng = np.random.RandomState(seed)
    n_samples = X.shape[0]
    bootstrap_size = int(n_samples * bootstrap_frac)

    references = {algo: _fit_algorithm(algo, X, ks[algo], seed=42) for algo in algorithms}

    rows = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=bootstrap_size, replace=True)
        X_boot = X[indices]
        row = {}
        for algo in algorithms:
            boot_labels = _fit_algorithm(algo, X_boot, ks[algo], seed=42)
            row[algo] = adjusted_rand_score(references[algo][indices], boot_labels)
        rows.append(row)

    return pd.DataFrame(rows)


def friedman_test(ari_df: pd.DataFrame) -> Dict[str, float]:
    """
    Teste de Friedman não paramétrico nas ARIs bootstrap pareadas
    (Demsar, 2006, JMLR).

    H0: as distribuições de ARI dos algoritmos são equivalentes.
    H1: pelo menos um algoritmo difere.

    Se p < 0.05, rejeita-se H0, e o pós-teste de Wilcoxon pareado com
    correção de Bonferroni identifica quais pares diferem.
    """
    columns = list(ari_df.columns)
    stat, p = friedmanchisquare(*[ari_df[c].values for c in columns])
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_blocks": len(ari_df),
        "n_treatments": len(columns),
        "means": {c: float(ari_df[c].mean()) for c in columns},
        "reject_h0": bool(p < 0.05),
    }


def wilcoxon_pairwise(ari_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Wilcoxon pareado entre todos os pares de algoritmos, com Bonferroni.

    Pós-teste do Friedman. Reporta p-valor bruto e p corrigido
    (Bonferroni: p_adj = p * n_pares).
    """
    cols = list(ari_df.columns)
    rows = []
    n_pairs = len(cols) * (len(cols) - 1) // 2
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            stat, p = wilcoxon(ari_df[a].values, ari_df[b].values)
            p_adj = min(p * n_pairs, 1.0)
            rows.append({
                "A": a,
                "B": b,
                "mean_A": float(ari_df[a].mean()),
                "mean_B": float(ari_df[b].mean()),
                "statistic": float(stat),
                "p_value": float(p),
                "p_bonferroni": float(p_adj),
                "significant": bool(p_adj < alpha),
            })
    return pd.DataFrame(rows)


def compare_algorithms(
    X: np.ndarray,
    results_dict: Dict,
    position_labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Tabela comparativa dos algoritmos: métricas internas, estabilidade
    (sementes e bootstrap) e, se houver posições, concordância com elas.
    """
    rows = []
    for algo in ["kmeans", "gmm", "hierarchical"]:
        result = results_dict[algo]
        internal = compute_internal_metrics(X, result.labels)
        print(f"  Calculando estabilidade para {algo}...")
        stability = compute_stability(X, algo, result.k)

        row = {
            "Algorithm": result.algorithm,
            "k": result.k,
            "Silhouette Score": internal["silhouette"],
            "Davies-Bouldin Index": internal["davies_bouldin"],
            "Calinski-Harabasz Index": internal["calinski_harabasz"],
            "Random Seed Mean ARI": stability["seed_mean_ari"],
            "Bootstrap Mean ARI": stability["bootstrap_mean_ari"],
            "Silhouette per Cluster": internal["silhouette_per_cluster"],
        }
        if position_labels is not None:
            external = compute_external_metrics(result.labels, position_labels)
            row["ARI vs Positions"] = external["ari"]
            row["NMI vs Positions"] = external["nmi"]
        rows.append(row)

    return pd.DataFrame(rows)


def crosstab_clusters_positions(
    labels: np.ndarray,
    position_labels: np.ndarray,
) -> pd.DataFrame:
    """Tabela de contingência: Guard/Forward/Center por cluster."""
    df_temp = pd.DataFrame({"Cluster": labels, "Position": position_labels})
    df_temp = df_temp[df_temp["Position"] != "Unknown"]
    return pd.crosstab(df_temp["Cluster"], df_temp["Position"], margins=True, margins_name="Total")
