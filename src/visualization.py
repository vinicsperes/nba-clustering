"""
Visualization Module
Creates plots and visualizations for clustering analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import FIGURES_DIR, VIZ_CONFIG


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
    method: str = "pca",
    player_names: List[str] = None,
    highlight_players: List[str] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot clusters in 2D using PCA or t-SNE.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    labels : np.ndarray
        Cluster labels
    title : str
        Plot title
    method : str
        Dimensionality reduction method ('pca' or 'tsne')
    player_names : List[str], optional
        Player names for annotation
    highlight_players : List[str], optional
        List of player names to highlight
    save_path : Path, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Dimensionality reduction
    if method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
        var_explained = reducer.explained_variance_ratio_
        xlabel = f"PC1 ({var_explained[0]*100:.1f}% variância)"
        ylabel = f"PC2 ({var_explained[1]*100:.1f}% variância)"
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        X_2d = reducer.fit_transform(X)
        xlabel = "t-SNE 1"
        ylabel = "t-SNE 2"

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

    # Highlight specific players
    if player_names and highlight_players:
        for i, name in enumerate(player_names):
            if name in highlight_players:
                ax.annotate(name, (X_2d[i, 0], X_2d[i, 1]),
                            fontsize=9, fontweight='bold',
                            xytext=(5, 5), textcoords='offset points')

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


def plot_cluster_profiles(
    cluster_profiles: pd.DataFrame,
    feature_columns: List[str],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot radar chart of cluster profiles.

    Parameters
    ----------
    cluster_profiles : pd.DataFrame
        DataFrame with cluster characteristics
    feature_columns : List[str]
        Feature columns to include in the radar
    save_path : Path, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    n_clusters = len(cluster_profiles)
    n_features = len(feature_columns)

    # Normalize features to 0-1 range for visualization
    profiles_norm = cluster_profiles[feature_columns].copy()
    for col in feature_columns:
        min_val = profiles_norm[col].min()
        max_val = profiles_norm[col].max()
        if max_val > min_val:
            profiles_norm[col] = (profiles_norm[col] - min_val) / (max_val - min_val)
        else:
            profiles_norm[col] = 0.5

    # Set up the radar chart
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

    colors = sns.color_palette(VIZ_CONFIG["palette"], n_clusters)

    for idx, row in profiles_norm.iterrows():
        values = row[feature_columns].tolist()
        values += values[:1]  # Close the polygon

        cluster_id = cluster_profiles.loc[idx, 'Cluster']
        ax.plot(angles, values, 'o-', linewidth=2,
                label=f'Cluster {cluster_id}', color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_columns, fontsize=10)
    ax.set_title('Perfil dos Clusters', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_cluster_comparison_heatmap(
    cluster_profiles: pd.DataFrame,
    feature_columns: List[str],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot heatmap comparing cluster profiles.

    Parameters
    ----------
    cluster_profiles : pd.DataFrame
        DataFrame with cluster characteristics
    feature_columns : List[str]
        Feature columns to include
    save_path : Path, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Prepare data
    data = cluster_profiles.set_index('Cluster')[feature_columns]

    # Standardize for better visualization
    data_std = (data - data.mean()) / data.std()

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        data_std.T,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        center=0,
        ax=ax,
        cbar_kws={'label': 'Desvios da Média'}
    )

    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Estatística', fontsize=12)
    ax.set_title('Comparação de Perfis dos Clusters (Padronizado)', fontsize=14)

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


def plot_algorithm_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot bar charts comparing algorithm performance.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with algorithm comparison metrics
    save_path : Path, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    metrics = ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"]
    available_metrics = [m for m in metrics if m in comparison_df.columns]

    fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 5))
    if len(available_metrics) == 1:
        axes = [axes]

    colors = sns.color_palette(VIZ_CONFIG["palette"], len(comparison_df))

    for ax, metric in zip(axes, available_metrics):
        bars = ax.bar(comparison_df["Algorithm"], comparison_df[metric], color=colors)
        ax.set_xlabel('Algoritmo', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(metric, fontsize=12)
        ax.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.suptitle('Comparação de Algoritmos de Clustering', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_stability_comparison(
    stability_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot stability comparison across algorithms.

    Parameters
    ----------
    stability_df : pd.DataFrame
        DataFrame with stability metrics
    save_path : Path, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("Random Seed Mean ARI", "ARI entre Seeds Aleatórias"),
        ("Bootstrap Mean ARI", "ARI Bootstrap"),
        ("Mean Point Stability", "Estabilidade por Ponto")
    ]

    colors = sns.color_palette(VIZ_CONFIG["palette"], len(stability_df))

    for ax, (metric, title) in zip(axes, metrics):
        if metric in stability_df.columns:
            bars = ax.bar(stability_df["Algorithm"], stability_df[metric], color=colors)
            ax.set_xlabel('Algoritmo', fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim([0, 1])

            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    plt.suptitle('Análise de Estabilidade dos Algoritmos', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def create_all_visualizations(
    X: np.ndarray,
    clustering_results: Dict[str, np.ndarray],
    analysis_results: Dict[str, Any],
    feature_names: List[str],
    player_names: List[str] = None,
    output_dir: Path = FIGURES_DIR
) -> None:
    """
    Create all visualizations and save to output directory.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    clustering_results : Dict[str, np.ndarray]
        Dictionary mapping algorithm name to labels
    analysis_results : Dict[str, Any]
        Results from analysis pipeline
    feature_names : List[str]
        Names of features
    player_names : List[str], optional
        Names of players
    output_dir : Path
        Directory to save figures
    """
    output_dir.mkdir(exist_ok=True)

    print("\nCreating visualizations...")

    # Correlation matrix of the clustering features (EDA)
    plot_correlation_matrix(
        X,
        feature_names,
        save_path=output_dir / "correlation_matrix.png"
    )

    # K-Means specific plots (elbow curve)
    if "kmeans" in analysis_results:
        kmeans_results = analysis_results["kmeans"]
        plot_elbow_curve(
            kmeans_results["k_values"],
            kmeans_results["inertias"],
            kmeans_results["silhouettes"],
            optimal_k=kmeans_results["optimal_k"],
            save_path=output_dir / "kmeans_elbow.png"
        )

    # GMM specific plots (BIC curve)
    if "gmm" in analysis_results:
        gmm_results = analysis_results["gmm"]
        plot_bic_curve(
            gmm_results["k_values"],
            gmm_results["bics"],
            gmm_results["silhouettes"],
            optimal_k=gmm_results["optimal_k"],
            save_path=output_dir / "gmm_bic.png"
        )

    # Hierarchical specific plots (dendrogram)
    if "hierarchical" in analysis_results:
        hier_results = analysis_results["hierarchical"]
        if "linkage_matrix" in hier_results and hier_results["linkage_matrix"] is not None:
            plot_dendrogram(
                hier_results["linkage_matrix"],
                n_clusters=hier_results.get("optimal_k"),
                save_path=output_dir / "hierarchical_dendrogram.png"
            )

    # Cluster visualizations for each algorithm
    for name, labels in clustering_results.items():
        # PCA visualization
        plot_clusters_2d(
            X, labels,
            title=f'{name} - Visualização PCA',
            method='pca',
            player_names=player_names,
            save_path=output_dir / f"{name.lower().replace(' ', '_')}_pca.png"
        )

        # Silhouette analysis
        plot_silhouette_analysis(
            X, labels,
            title=f'{name} - Análise de Silhouette',
            save_path=output_dir / f"{name.lower().replace(' ', '_')}_silhouette.png"
        )

    # Comparison plots
    if "comparison" in analysis_results:
        plot_algorithm_comparison(
            analysis_results["comparison"],
            save_path=output_dir / "algorithm_comparison.png"
        )

    if "stability" in analysis_results:
        plot_stability_comparison(
            analysis_results["stability"],
            save_path=output_dir / "stability_comparison.png"
        )

    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    # Test visualizations with sample data
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans

    # Generate sample data
    X, _ = make_blobs(n_samples=200, centers=4, random_state=42)

    # Run K-means
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X)

    # Test PCA visualization
    fig = plot_clusters_2d(X, labels, title="Test Clusters")
    plt.show()

    # Test silhouette analysis
    fig = plot_silhouette_analysis(X, labels)
    plt.show()
