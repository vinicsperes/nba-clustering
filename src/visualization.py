"""
Módulo de visualização: gera as figuras usadas na monografia.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
from typing import Dict, List, Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import VIZ_CONFIG


# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(VIZ_CONFIG["palette"])


def plot_elbow_curve(
    k_values: List[int],
    inertias: List[float],
    silhouettes: List[float],
    optimal_k: int = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot elbow curve for K-means optimization.

    Parameters
    ----------
    k_values : List[int]
        List of k values tested
    inertias : List[float]
        Inertia for each k
    silhouettes : List[float]
        Silhouette score for each k
    optimal_k : int, optional
        Optimal k value to highlight
    save_path : Path, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow curve (Inertia)
    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Número de Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inércia', fontsize=12)
    ax1.set_title('Método do Cotovelo', fontsize=14)
    ax1.grid(True, alpha=0.3)

    if optimal_k:
        idx = k_values.index(optimal_k)
        ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'k ótimo = {optimal_k}')
        ax1.scatter([optimal_k], [inertias[idx]], color='r', s=200, zorder=5)
        ax1.legend()

    # Silhouette score
    ax2.plot(k_values, silhouettes, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Número de Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score por k', fontsize=14)
    ax2.grid(True, alpha=0.3)

    if optimal_k:
        idx = k_values.index(optimal_k)
        ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'k ótimo = {optimal_k}')
        ax2.scatter([optimal_k], [silhouettes[idx]], color='r', s=200, zorder=5)
        ax2.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig



def plot_bic_curve(
    k_values: List[int],
    bics: List[float],
    silhouettes: List[float],
    optimal_k: int = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot BIC curve for GMM component selection.

    Parameters
    ----------
    k_values : List[int]
        List of k values tested
    bics : List[float]
        BIC for each k (lower is better)
    silhouettes : List[float]
        Silhouette score for each k (informational only)
    optimal_k : int, optional
        Optimal k value to highlight
    save_path : Path, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # BIC curve (lower is better)
    ax1.plot(k_values, bics, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Número de Componentes (k)', fontsize=12)
    ax1.set_ylabel('BIC', fontsize=12)
    ax1.set_title('Critério de Informação Bayesiano (BIC)', fontsize=14)
    ax1.grid(True, alpha=0.3)

    if optimal_k:
        idx = k_values.index(optimal_k)
        ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'k ótimo = {optimal_k}')
        ax1.scatter([optimal_k], [bics[idx]], color='r', s=200, zorder=5)
        ax1.legend()

    # Silhouette score (informational, not used for selection)
    ax2.plot(k_values, silhouettes, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Número de Componentes (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score por k (informativo)', fontsize=14)
    ax2.grid(True, alpha=0.3)

    if optimal_k:
        idx = k_values.index(optimal_k)
        ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'k ótimo = {optimal_k}')
        ax2.scatter([optimal_k], [silhouettes[idx]], color='r', s=200, zorder=5)
        ax2.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_correlation_matrix(
    X: np.ndarray,
    feature_names: List[str],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot correlation matrix heatmap of the clustering features.

    Correlation is scale-invariant, so standardized or raw features
    produce the same matrix.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    feature_names : List[str]
        Names of features
    save_path : Path, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    corr = pd.DataFrame(X, columns=feature_names).corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlação de Pearson'},
        ax=ax,
    )
    ax.set_title('Matriz de Correlação das Features', fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    n_clusters: int = None,
    labels: List[str] = None,
    truncate_mode: str = 'lastp',
    p: int = 30,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot hierarchical clustering dendrogram.

    Parameters
    ----------
    linkage_matrix : np.ndarray
        Linkage matrix from scipy hierarchical clustering
    n_clusters : int, optional
        Number of clusters (for color threshold)
    labels : List[str], optional
        Labels for leaf nodes
    truncate_mode : str
        Truncation mode for large dendrograms
    p : int
        Number of leaves to show when truncating
    save_path : Path, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Calculate color threshold if n_clusters specified
    if n_clusters:
        color_threshold = linkage_matrix[-(n_clusters - 1), 2]
    else:
        color_threshold = None

    dendrogram(
        linkage_matrix,
        ax=ax,
        truncate_mode=truncate_mode,
        p=p,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=color_threshold,
        labels=labels
    )

    ax.set_xlabel('Amostras', fontsize=12)
    ax.set_ylabel('Distância', fontsize=12)
    ax.set_title('Dendrograma - Clustering Hierárquico', fontsize=14)

    if n_clusters:
        ax.axhline(y=color_threshold, color='r', linestyle='--',
                   label=f'Corte para {n_clusters} clusters')
        ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_clusters_2d(
    X: np.ndarray,
    labels: np.ndarray,
    title: str = "Visualização dos Clusters",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot clusters in 2D using the first two principal components (PCA).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    labels : np.ndarray
        Cluster labels
    title : str
        Plot title
    save_path : Path, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    reducer = PCA(n_components=2, random_state=42)
    X_2d = reducer.fit_transform(X)
    var_explained = reducer.explained_variance_ratio_
    xlabel = f"PC1 ({var_explained[0]*100:.1f}% variância)"
    ylabel = f"PC2 ({var_explained[1]*100:.1f}% variância)"

    fig, ax = plt.subplots(figsize=(12, 10))

    # Get unique labels (excluding noise)
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l >= 0])

    # Color palette
    colors = sns.color_palette(VIZ_CONFIG["palette"], n_clusters)
    color_map = {label: colors[i] for i, label in enumerate(sorted([l for l in unique_labels if l >= 0]))}
    color_map[-1] = (0.5, 0.5, 0.5)  # Gray for noise

    # Plot each cluster
    for label in unique_labels:
        mask = labels == label
        color = color_map[label]

        if label == -1:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[color], marker='x', s=50, alpha=0.5,
                       label='Outliers')
        else:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[color], s=80, alpha=0.7,
                       label=f'Cluster {label}')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_silhouette_analysis(
    X: np.ndarray,
    labels: np.ndarray,
    title: str = "Análise de Silhouette",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot silhouette analysis showing per-sample silhouette values.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    labels : np.ndarray
        Cluster labels
    title : str
        Plot title
    save_path : Path, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    from sklearn.metrics import silhouette_samples, silhouette_score

    # Filter out noise points
    mask = labels >= 0
    X_valid = X[mask]
    labels_valid = labels[mask]

    n_clusters = len(set(labels_valid))
    silhouette_avg = silhouette_score(X_valid, labels_valid)
    sample_silhouette_values = silhouette_samples(X_valid, labels_valid)

    fig, ax = plt.subplots(figsize=(10, 8))

    y_lower = 10
    colors = sns.color_palette(VIZ_CONFIG["palette"], n_clusters)

    for i, cluster in enumerate(sorted(set(labels_valid))):
        cluster_silhouette_values = sample_silhouette_values[labels_valid == cluster]
        cluster_silhouette_values.sort()

        size_cluster = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouette_values,
            facecolor=colors[i],
            edgecolor=colors[i],
            alpha=0.7
        )

        ax.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster), fontsize=10)
        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color="red", linestyle="--",
               label=f'Silhouette médio = {silhouette_avg:.3f}')

    ax.set_xlabel('Silhouette Coefficient', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.set_xlim([-0.1, 1])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def create_figures(
    X: np.ndarray,
    clustering: Dict[str, object],
    feature_names: List[str],
    output_dir: Path,
) -> None:
    """
    Gera as oito figuras da monografia em output_dir.

    `clustering` é o dicionário retornado por `run_clustering`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    kmeans, gmm, hier = clustering["kmeans"], clustering["gmm"], clustering["hierarchical"]

    plot_correlation_matrix(X, feature_names, save_path=output_dir / "correlation_matrix.png")

    elbow = clustering["kmeans_model"].get_elbow_data()
    plot_elbow_curve(
        elbow["k"].tolist(), elbow["inertia"].tolist(), elbow["silhouette"].tolist(),
        optimal_k=kmeans.k, save_path=output_dir / "kmeans_elbow.png",
    )

    bic = clustering["gmm_model"].get_bic_data()
    plot_bic_curve(
        bic["k"].tolist(), bic["bic"].tolist(), bic["silhouette"].tolist(),
        optimal_k=gmm.k, save_path=output_dir / "gmm_bic.png",
    )

    plot_dendrogram(
        hier.linkage_matrix, n_clusters=hier.k,
        save_path=output_dir / "hierarchical_dendrogram.png",
    )

    for name, result in [("K-Means", kmeans), ("GMM", gmm), ("Hierarchical", hier)]:
        plot_clusters_2d(
            X, result.labels, title=f"{name} - Visualização PCA",
            save_path=output_dir / f"{name.lower()}_pca.png",
        )

    plot_silhouette_analysis(
        X, kmeans.labels, title="K-Means - Análise de Silhouette",
        save_path=output_dir / "k-means_silhouette.png",
    )

    plt.close("all")
