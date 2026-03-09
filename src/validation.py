"""
Módulo de validação científica dos clusters.

Métricas calculadas:
1. Internas: Silhouette Score, Calinski-Harabasz, Davies-Bouldin
2. Estabilidade: Multi-seed ARI (30 seeds), Bootstrap ARI (100 iterações)
3. Externa (secundária): ARI e NMI vs posições tradicionais (Guard/Forward/Center)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

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


@dataclass
class ValidationResult:
    """Resultado completo da validação."""
    # Métricas internas
    silhouette: float
    silhouette_per_cluster: Dict[int, float]
    calinski_harabasz: float
    davies_bouldin: float

    # Validação externa (vs posições)
    ari_vs_positions: float
    nmi_vs_positions: float

    # Comparação de variância
    cluster_variance: float
    position_variance: float
    variance_ratio: float
    clusters_more_homogeneous: bool

    # Interpretação
    hypothesis_supported: bool
    evidence_summary: str


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

    Parameters
    ----------
    X : np.ndarray
        Dados normalizados
    labels : np.ndarray
        Labels dos clusters

    Returns
    -------
    Dict[str, float]
        Métricas calculadas
    """
    mask = labels != -1
    X_clean = X[mask]
    labels_clean = labels[mask]

    sil = silhouette_score(X_clean, labels_clean)
    sil_samples = silhouette_samples(X_clean, labels_clean)

    sil_per_cluster = {
        int(c): float(sil_samples[labels_clean == c].mean())
        for c in np.unique(labels_clean)
    }

    ch = calinski_harabasz_score(X_clean, labels_clean)
    db = davies_bouldin_score(X_clean, labels_clean)

    return {
        "silhouette": sil,
        "silhouette_per_cluster": sil_per_cluster,
        "calinski_harabasz": ch,
        "davies_bouldin": db
    }


def compute_external_metrics(
    cluster_labels: np.ndarray,
    position_labels: np.ndarray
) -> Dict[str, float]:
    """
    Calcula métricas de validação externa contra posições tradicionais.

    ARI mede concordância entre clusters e posições (análise secundária).
    ARI ~0 = independentes, ARI ~1 = idênticos.

    Parameters
    ----------
    cluster_labels : np.ndarray
        Labels dos clusters
    position_labels : np.ndarray
        Labels das posições tradicionais (G/F/C)

    Returns
    -------
    Dict[str, float]
        Métricas de concordância
    """
    known_mask = position_labels != "Unknown"
    if known_mask.sum() < 10:
        print("AVISO: poucas posições conhecidas para validação externa")
        return {"ari": 0, "nmi": 0}

    cluster_clean = cluster_labels[known_mask]
    position_clean = position_labels[known_mask]

    ari = adjusted_rand_score(position_clean, cluster_clean)
    nmi = normalized_mutual_info_score(position_clean, cluster_clean)

    return {
        "ari": ari,
        "nmi": nmi,
        "n_samples": known_mask.sum()
    }


def compute_variance_comparison(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    position_labels: np.ndarray
) -> Dict[str, float]:
    """
    Compara variância intra-grupo: clusters vs posições tradicionais.

    Se var(clusters) < var(posições), clusters agrupam jogadores
    mais similares entre si do que a classificação posicional.

    Parameters
    ----------
    X : np.ndarray
        Dados normalizados
    cluster_labels : np.ndarray
        Labels dos clusters
    position_labels : np.ndarray
        Labels das posições

    Returns
    -------
    Dict[str, float]
        Comparação de variâncias
    """
    def mean_intra_variance(X, labels):
        """Variância média dentro dos grupos."""
        unique = np.unique(labels)
        variances = []
        sizes = []

        for label in unique:
            if label in ["Unknown", -1, "-1"]:
                continue
            mask = labels == label
            n = mask.sum()
            if n > 1:
                # Variância média de todas as features
                var = np.mean(np.var(X[mask], axis=0))
                variances.append(var)
                sizes.append(n)

        if not variances:
            return 0, {}

        # Média ponderada por tamanho do grupo
        total = sum(sizes)
        weighted_var = sum(v * s for v, s in zip(variances, sizes)) / total

        return weighted_var, dict(zip(unique, variances))

    cluster_var, cluster_vars = mean_intra_variance(X, cluster_labels)
    position_var, position_vars = mean_intra_variance(X, position_labels)

    ratio = cluster_var / max(position_var, 0.001)

    return {
        "cluster_variance": cluster_var,
        "position_variance": position_var,
        "variance_ratio": ratio,
        "clusters_more_homogeneous": cluster_var < position_var,
        "improvement_pct": (1 - ratio) * 100 if ratio < 1 else 0,
        "cluster_variances_detail": cluster_vars,
        "position_variances_detail": position_vars
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
    Calcula estabilidade de um algoritmo de clustering.

    Multi-seed: roda com n_seeds diferentes, calcula ARI entre pares.
    Bootstrap: amostra bootstrap_frac dos dados n_bootstrap vezes, calcula ARI médio.
    Hierárquico: pula multi-seed (determinístico), faz só bootstrap.

    Parameters
    ----------
    X : np.ndarray
        Dados normalizados
    algorithm : str
        Nome do algoritmo: "kmeans", "gmm", "hierarchical"
    k : int
        Número de clusters
    n_seeds : int
        Número de seeds para multi-seed
    bootstrap_frac : float
        Fração de dados para bootstrap
    n_bootstrap : int
        Número de iterações bootstrap

    Returns
    -------
    Dict[str, float]
        seed_mean_ari, bootstrap_mean_ari, point_stability
    """
    rng = np.random.RandomState(42)
    n_samples = X.shape[0]

    # --- Multi-seed stability ---
    seed_mean_ari = 1.0  # Determinístico = perfeito
    if algorithm != "hierarchical":
        all_labels = []
        seeds = rng.randint(0, 100000, size=n_seeds)
        for seed in seeds:
            labels = _fit_algorithm(algorithm, X, k, seed=int(seed))
            all_labels.append(labels)

        aris = []
        for i in range(len(all_labels)):
            for j in range(i + 1, len(all_labels)):
                aris.append(adjusted_rand_score(all_labels[i], all_labels[j]))
        seed_mean_ari = float(np.mean(aris)) if aris else 1.0

    # --- Bootstrap stability ---
    bootstrap_size = int(n_samples * bootstrap_frac)
    reference_labels = _fit_algorithm(algorithm, X, k, seed=42)

    bootstrap_aris = []
    point_counts = np.zeros(n_samples)
    point_same = np.zeros(n_samples)

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=bootstrap_size, replace=True)
        X_boot = X[indices]
        boot_labels = _fit_algorithm(algorithm, X_boot, k, seed=42)

        # Compute ARI on the bootstrap sample
        ref_sub = reference_labels[indices]
        ari = adjusted_rand_score(ref_sub, boot_labels)
        bootstrap_aris.append(ari)

        # Track point-level stability
        for idx_boot, idx_orig in enumerate(indices):
            point_counts[idx_orig] += 1
            if boot_labels[idx_boot] == reference_labels[idx_orig]:
                point_same[idx_orig] += 1

    bootstrap_mean_ari = float(np.mean(bootstrap_aris))

    # Point stability: fraction of times each player stayed in same cluster
    with np.errstate(divide='ignore', invalid='ignore'):
        point_stab = np.where(point_counts > 0, point_same / point_counts, 0.0)
    mean_point_stability = float(np.mean(point_stab))

    return {
        "seed_mean_ari": seed_mean_ari,
        "bootstrap_mean_ari": bootstrap_mean_ari,
        "point_stability": mean_point_stability,
    }


def validate_clustering(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    position_labels: np.ndarray,
    algorithm_name: str = "K-Means"
) -> ValidationResult:
    """
    Validação científica de um resultado de clustering.

    Calcula métricas internas e externas para avaliar a qualidade
    dos agrupamentos produzidos por um algoritmo.

    Parameters
    ----------
    X : np.ndarray
        Dados normalizados
    cluster_labels : np.ndarray
        Labels dos clusters
    position_labels : np.ndarray
        Labels das posições tradicionais
    algorithm_name : str
        Nome do algoritmo (para exibição)

    Returns
    -------
    ValidationResult
        Resultado completo com métricas
    """
    print(f"\n{'=' * 60}")
    print(f"VALIDAÇÃO - {algorithm_name}")
    print("=" * 60)

    # 1. Métricas internas
    print("\n[1] Métricas Internas")
    internal = compute_internal_metrics(X, cluster_labels)
    print(f"    Silhouette Score:  {internal['silhouette']:.4f}")
    print(f"    Calinski-Harabasz: {internal['calinski_harabasz']:.2f}")
    print(f"    Davies-Bouldin:    {internal['davies_bouldin']:.4f}")

    # Interpretação objetiva do Silhouette
    sil = internal['silhouette']
    if sil >= 0.50:
        sil_interp = "forte"
    elif sil >= 0.25:
        sil_interp = "razoável"
    else:
        sil_interp = "fraca"
    print(f"    Interpretação: estrutura {sil_interp} (Silhouette {sil:.3f})")

    # 2. Validação externa (análise secundária)
    print("\n[2] Concordância com Posições Tradicionais (análise secundária)")
    external = compute_external_metrics(cluster_labels, position_labels)
    print(f"    Adjusted Rand Index: {external['ari']:.4f}")
    print(f"    Normalized MI:       {external['nmi']:.4f}")

    if external['ari'] < 0.10:
        print("    Interpretação: clusters independentes das posições")
    elif external['ari'] < 0.30:
        print("    Interpretação: sobreposição parcial com posições")
    else:
        print("    Interpretação: concordância moderada com posições")

    # 3. Comparação de variância
    print("\n[3] Homogeneidade Intra-Grupo")
    variance = compute_variance_comparison(X, cluster_labels, position_labels)
    print(f"    Variância intra-cluster: {variance['cluster_variance']:.4f}")
    print(f"    Variância intra-posição: {variance['position_variance']:.4f}")
    print(f"    Ratio (cluster/posição): {variance['variance_ratio']:.4f}")

    if variance['clusters_more_homogeneous']:
        print(f"    Clusters {variance['improvement_pct']:.1f}% mais homogêneos que posições")
    else:
        print(f"    Posições mais homogêneas que clusters")

    # Gerar sumário descritivo
    summary = generate_evidence_summary(internal, external, variance)

    quality_ok = sil >= 0.20 and variance['clusters_more_homogeneous']

    print(f"\n{'=' * 60}")

    return ValidationResult(
        silhouette=internal['silhouette'],
        silhouette_per_cluster=internal['silhouette_per_cluster'],
        calinski_harabasz=internal['calinski_harabasz'],
        davies_bouldin=internal['davies_bouldin'],
        ari_vs_positions=external['ari'],
        nmi_vs_positions=external['nmi'],
        cluster_variance=variance['cluster_variance'],
        position_variance=variance['position_variance'],
        variance_ratio=variance['variance_ratio'],
        clusters_more_homogeneous=variance['clusters_more_homogeneous'],
        hypothesis_supported=quality_ok,
        evidence_summary=summary
    )


def generate_evidence_summary(
    internal: Dict,
    external: Dict,
    variance: Dict,
) -> str:
    """Gera texto descritivo resumindo as métricas de validação."""
    sil = internal['silhouette']
    db = internal['davies_bouldin']
    ch = internal['calinski_harabasz']
    ari = external['ari']
    var_ratio = variance['variance_ratio']
    improvement = variance.get('improvement_pct', 0)

    summary = f"""
MÉTRICAS DE QUALIDADE:
- Silhouette Score: {sil:.4f} (coesão e separação dos clusters)
- Davies-Bouldin Index: {db:.4f} (similaridade inter-cluster, menor = melhor)
- Calinski-Harabasz Index: {ch:.2f} (razão de dispersão, maior = melhor)

CONCORDÂNCIA COM POSIÇÕES (análise secundária):
- ARI: {ari:.4f} (concordância ajustada com posições tradicionais)
- Os clusters capturam padrões de desempenho que transcendem a
  classificação posicional tradicional (Guard/Forward/Center).

HOMOGENEIDADE:
- Variância intra-cluster: {variance['cluster_variance']:.4f}
- Variância intra-posição: {variance['position_variance']:.4f}
- Ratio: {var_ratio:.4f}"""

    if variance['clusters_more_homogeneous']:
        summary += f"""
- Os clusters são {improvement:.1f}% mais homogêneos que as posições,
  indicando agrupamentos estatisticamente mais coerentes."""
    else:
        summary += """
- As posições tradicionais apresentam menor variância intra-grupo."""

    return summary


def generate_validation_report(
    result: ValidationResult,
    cluster_profiles: pd.DataFrame,
    algorithm_name: str = "K-Means",
    output_path: Optional[Path] = None
) -> str:
    """
    Gera relatório de validação em formato texto para um algoritmo.
    """
    report = f"""
================================================================================
RELATÓRIO DE VALIDAÇÃO - {algorithm_name}
Análise Comparativa de Algoritmos de Clustering - NBA Player Archetypes
================================================================================

1. MÉTRICAS DE QUALIDADE INTERNA
--------------------------------------------------------------------------------
Silhouette Score:     {result.silhouette:.4f}
Calinski-Harabasz:    {result.calinski_harabasz:.2f}
Davies-Bouldin:       {result.davies_bouldin:.4f}

Silhouette por Cluster:
{pd.Series(result.silhouette_per_cluster).to_string()}

2. CONCORDÂNCIA COM POSIÇÕES TRADICIONAIS (análise secundária)
--------------------------------------------------------------------------------
Adjusted Rand Index:  {result.ari_vs_positions:.4f}
Normalized MI:        {result.nmi_vs_positions:.4f}

3. HOMOGENEIDADE INTRA-GRUPO
--------------------------------------------------------------------------------
Variância Intra-Cluster:  {result.cluster_variance:.6f}
Variância Intra-Posição:  {result.position_variance:.6f}
Ratio (cluster/posição):  {result.variance_ratio:.4f}

Clusters mais homogêneos: {"SIM" if result.clusters_more_homogeneous else "NÃO"}

4. PERFIS DOS CLUSTERS
--------------------------------------------------------------------------------
{cluster_profiles.to_string()}

5. RESUMO
--------------------------------------------------------------------------------
{result.evidence_summary}

================================================================================
"""

    if output_path:
        output_path.write_text(report)
        print(f"Relatório salvo em: {output_path}")

    return report


def compare_algorithms(
    X: np.ndarray,
    results_dict: Dict,
    position_labels: np.ndarray,
) -> pd.DataFrame:
    """
    Tabela comparativa dos algoritmos de clustering.

    Compara: Silhouette, Davies-Bouldin, Calinski-Harabasz,
    ARI vs posições, estabilidade.

    Parameters
    ----------
    X : np.ndarray
        Dados normalizados
    results_dict : Dict
        Dicionário com ClusteringResult por algoritmo
    position_labels : np.ndarray
        Labels das posições tradicionais

    Returns
    -------
    pd.DataFrame
        DataFrame com uma linha por algoritmo
    """
    print("\n" + "=" * 60)
    print("COMPARAÇÃO DE ALGORITMOS")
    print("=" * 60)

    rows = []
    algorithms = ["kmeans", "gmm", "hierarchical"]

    for algo in algorithms:
        if algo not in results_dict:
            continue

        result = results_dict[algo]
        labels = result.labels

        # Métricas internas
        internal = compute_internal_metrics(X, labels)

        # Métricas externas
        external = compute_external_metrics(labels, position_labels)

        # Estabilidade
        print(f"\n  Calculando estabilidade para {algo}...")
        stability = compute_stability(X, algo, result.k)

        row = {
            "Algorithm": result.algorithm,
            "k": result.k,
            "Silhouette Score": internal["silhouette"],
            "Davies-Bouldin Index": internal["davies_bouldin"],
            "Calinski-Harabasz Index": internal["calinski_harabasz"],
            "ARI vs Positions": external["ari"],
            "NMI vs Positions": external["nmi"],
            "Random Seed Mean ARI": stability["seed_mean_ari"],
            "Bootstrap Mean ARI": stability["bootstrap_mean_ari"],
            "Mean Point Stability": stability["point_stability"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Mostrar tabela
    print("\n" + df.to_string(index=False))

    return df


def crosstab_clusters_positions(
    labels: np.ndarray,
    position_labels: np.ndarray,
    algorithm_name: str = "",
) -> pd.DataFrame:
    """
    Crosstab mostrando distribuição de Guard/Forward/Center por cluster.

    Parameters
    ----------
    labels : np.ndarray
        Labels dos clusters
    position_labels : np.ndarray
        Labels das posições (Guard, Forward, Center, Unknown)
    algorithm_name : str
        Nome do algoritmo para exibição

    Returns
    -------
    pd.DataFrame
        Tabela de contingência
    """
    df_temp = pd.DataFrame({
        "Cluster": labels,
        "Position": position_labels,
    })

    # Filtrar Unknown
    df_temp = df_temp[df_temp["Position"] != "Unknown"]

    ct = pd.crosstab(df_temp["Cluster"], df_temp["Position"], margins=True)

    if algorithm_name:
        print(f"\n  Crosstab - {algorithm_name}:")
    print(ct.to_string())

    return ct


