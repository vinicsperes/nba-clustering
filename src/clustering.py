"""
Módulo de clustering.

Implementa os três algoritmos comparados no TCC:
- K-Means: hard clustering, baseline da literatura
- GMM (Gaussian Mixture Models): soft clustering, captura jogadores híbridos
- Clustering Hierárquico Aglomerativo: determinístico, gera dendrograma
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import KMEANS_CONFIG, GMM_CONFIG, HIERARCHICAL_CONFIG


@dataclass
class ClusteringResult:
    """Resultado de um algoritmo de clustering."""
    algorithm: str
    k: int
    labels: np.ndarray
    silhouette: float
    inertia: Optional[float] = None          # K-Means
    bic: Optional[float] = None              # GMM
    probabilities: Optional[np.ndarray] = None  # GMM
    linkage_matrix: Optional[np.ndarray] = None  # Hierárquico


class KMeansClusterer:
    """
    K-Means clustering.

    Testa todos os valores de k no range configurado e seleciona
    o melhor pelo Silhouette Score. Serve como algoritmo baseline.
    """

    def __init__(
        self,
        k_range: range = KMEANS_CONFIG["k_range"],
        n_init: int = KMEANS_CONFIG["n_init"],
        max_iter: int = KMEANS_CONFIG["max_iter"],
        random_state: int = KMEANS_CONFIG["random_state"],
    ):
        self.k_range = k_range
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.best_k: Optional[int] = None
        self._results: Dict[int, ClusteringResult] = {}
        self._best_model: Optional[KMeans] = None

    def fit(self, X: np.ndarray) -> "KMeansClusterer":
        print("\n" + "=" * 50)
        print("K-MEANS")
        print("=" * 50)

        best_sil = -1
        for k in self.k_range:
            model = KMeans(
                n_clusters=k,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            labels = model.fit_predict(X)
            sil = silhouette_score(X, labels)

            self._results[k] = ClusteringResult(
                algorithm="K-Means",
                k=k,
                labels=labels,
                silhouette=sil,
                inertia=model.inertia_,
            )
            print(f"  k={k:2d}: Silhouette={sil:.3f}, Inertia={model.inertia_:.1f}")

            if sil > best_sil:
                best_sil = sil
                self.best_k = k
                self._best_model = model

        print(f"\n  Melhor k: {self.best_k} (Silhouette={best_sil:.3f})")
        return self

    def get_best_result(self) -> ClusteringResult:
        if self.best_k is None:
            raise ValueError("Modelo não ajustado. Use fit() primeiro.")
        return self._results[self.best_k]

    def get_elbow_data(self) -> pd.DataFrame:
        """Retorna dados de inércia e silhouette por k para o gráfico de cotovelo."""
        return pd.DataFrame({
            "k": list(self._results.keys()),
            "inertia": [r.inertia for r in self._results.values()],
            "silhouette": [r.silhouette for r in self._results.values()],
        })


class GMMClusterer:
    """
    Gaussian Mixture Models — soft clustering.

    Atribui probabilidades de pertencimento a cada cluster, permitindo
    identificar jogadores híbridos com perfil em múltiplos arquétipos.

    A seleção do número de componentes é feita pelo BIC por padrão
    (recomendação de Géron, 2025, p. 275: o Silhouette não é confiável
    para GMM porque os clusters não são necessariamente esféricos).
    """

    def __init__(
        self,
        k_range: range = GMM_CONFIG["k_range"],
        covariance_type: str = GMM_CONFIG["covariance_type"],
        n_init: int = GMM_CONFIG["n_init"],
        max_iter: int = GMM_CONFIG["max_iter"],
        random_state: int = GMM_CONFIG["random_state"],
        selection_metric: str = "bic",
    ):
        if selection_metric not in {"bic", "silhouette"}:
            raise ValueError(
                f"selection_metric deve ser 'bic' ou 'silhouette', recebido: {selection_metric}"
            )
        self.k_range = k_range
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.selection_metric = selection_metric
        self.best_k: Optional[int] = None
        self._results: Dict[int, ClusteringResult] = {}

    def get_bic_data(self) -> pd.DataFrame:
        """Retorna BIC e Silhouette por k para seleção de número de componentes."""
        return pd.DataFrame({
            "k": list(self._results.keys()),
            "bic": [r.bic for r in self._results.values()],
            "silhouette": [r.silhouette for r in self._results.values()],
        })

    def fit(self, X: np.ndarray) -> "GMMClusterer":
        print("\n" + "=" * 50)
        print(f"GMM (Gaussian Mixture Models) — selection_metric={self.selection_metric}")
        print("=" * 50)

        for k in self.k_range:
            model = GaussianMixture(
                n_components=k,
                covariance_type=self.covariance_type,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            model.fit(X)
            labels = model.predict(X)
            probs = model.predict_proba(X)
            bic = model.bic(X)
            sil = silhouette_score(X, labels)

            self._results[k] = ClusteringResult(
                algorithm="GMM",
                k=k,
                labels=labels,
                silhouette=sil,
                bic=bic,
                probabilities=probs,
            )
            print(f"  k={k:2d}: Silhouette={sil:.3f}, BIC={bic:.1f}")

        if self.selection_metric == "bic":
            self.best_k = min(self._results, key=lambda k: self._results[k].bic)
            print(
                f"\n  Melhor k: {self.best_k} "
                f"(BIC={self._results[self.best_k].bic:.1f}, "
                f"Silhouette={self._results[self.best_k].silhouette:.3f})"
            )
        else:
            self.best_k = max(self._results, key=lambda k: self._results[k].silhouette)
            print(
                f"\n  Melhor k: {self.best_k} "
                f"(Silhouette={self._results[self.best_k].silhouette:.3f}, "
                f"BIC={self._results[self.best_k].bic:.1f})"
            )
        return self

    def get_best_result(self) -> ClusteringResult:
        if self.best_k is None:
            raise ValueError("Modelo não ajustado. Use fit() primeiro.")
        return self._results[self.best_k]


class HierarchicalClusterer:
    """
    Clustering Hierárquico Aglomerativo (Ward linkage).

    Algoritmo determinístico que constrói uma hierarquia de clusters.
    Gera a linkage matrix necessária para o dendrograma.
    """

    def __init__(
        self,
        k_range: range = HIERARCHICAL_CONFIG["k_range"],
        linkage_method: str = HIERARCHICAL_CONFIG["linkage"],
    ):
        self.k_range = k_range
        self.linkage_method = linkage_method
        self.best_k: Optional[int] = None
        self._results: Dict[int, ClusteringResult] = {}
        self.linkage_matrix_: Optional[np.ndarray] = None

    def get_silhouette_data(self) -> pd.DataFrame:
        """Retorna Silhouette Score por k."""
        return pd.DataFrame({
            "k": list(self._results.keys()),
            "silhouette": [r.silhouette for r in self._results.values()],
        })

    def fit(self, X: np.ndarray) -> "HierarchicalClusterer":
        print("\n" + "=" * 50)
        print(f"CLUSTERING HIERÁRQUICO (linkage={self.linkage_method})")
        print("=" * 50)

        self.linkage_matrix_ = linkage(X, method=self.linkage_method)

        best_sil = -1
        for k in self.k_range:
            model = AgglomerativeClustering(n_clusters=k, linkage=self.linkage_method)
            labels = model.fit_predict(X)
            sil = silhouette_score(X, labels)

            self._results[k] = ClusteringResult(
                algorithm="Hierarchical",
                k=k,
                labels=labels,
                silhouette=sil,
                linkage_matrix=self.linkage_matrix_,
            )
            print(f"  k={k:2d}: Silhouette={sil:.3f}")

            if sil > best_sil:
                best_sil = sil
                self.best_k = k

        print(f"\n  Melhor k: {self.best_k} (Silhouette={best_sil:.3f})")
        return self

    def get_best_result(self) -> ClusteringResult:
        if self.best_k is None:
            raise ValueError("Modelo não ajustado. Use fit() primeiro.")
        return self._results[self.best_k]


def run_clustering(X: np.ndarray) -> Dict[str, object]:
    """
    Executa os três algoritmos de clustering e retorna os melhores resultados.

    Returns
    -------
    dict com chaves: "kmeans", "gmm", "hierarchical" (ClusteringResult),
    "kmeans_model" (KMeansClusterer, necessário para os dados do elbow)
    e "gmm_model" (GMMClusterer, necessário para os dados do BIC).
    """
    kmeans = KMeansClusterer()
    kmeans.fit(X)

    gmm = GMMClusterer()
    gmm.fit(X)

    hier = HierarchicalClusterer()
    hier.fit(X)

    return {
        "kmeans": kmeans.get_best_result(),
        "gmm": gmm.get_best_result(),
        "hierarchical": hier.get_best_result(),
        "kmeans_model": kmeans,
        "gmm_model": gmm,
    }


def get_cluster_profiles(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    df_original: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula o perfil médio de cada cluster na escala original das features.
    """
    profiles = []
    for cluster in np.unique(labels):
        if cluster == -1:
            continue
        mask = labels == cluster
        profile = {"Cluster": cluster, "N_Players": int(mask.sum())}
        for feat in feature_names:
            if feat in df_original.columns:
                profile[feat] = df_original.loc[mask, feat].mean()
        profiles.append(profile)
    return pd.DataFrame(profiles)


def get_cluster_examples(
    labels: np.ndarray,
    player_names: List[str],
    n_examples: int = 5,
) -> Dict[int, List[str]]:
    """Retorna exemplos de jogadores por cluster."""
    examples = {}
    for cluster in np.unique(labels):
        if cluster == -1:
            continue
        indices = np.where(labels == cluster)[0]
        examples[cluster] = [player_names[i] for i in indices[:n_examples]]
    return examples


def analyze_hybrid_players(
    gmm_probs: np.ndarray,
    player_names: List[str],
    threshold: float = 0.25,
) -> pd.DataFrame:
    """
    Identifica jogadores híbridos: aqueles com probabilidade significativa
    (acima de `threshold`) em um cluster secundário no modelo GMM.

    Esses jogadores ilustram o conceito de "positionless basketball".
    """
    hybrids = []
    for name, probs in zip(player_names, gmm_probs):
        sorted_idx = np.argsort(probs)[::-1]
        primary, secondary = sorted_idx[0], sorted_idx[1]
        if probs[secondary] >= threshold:
            hybrids.append({
                "Player": name,
                "Primary_Cluster": int(primary),
                "Primary_Prob": float(probs[primary]),
                "Secondary_Cluster": int(secondary),
                "Secondary_Prob": float(probs[secondary]),
                "Hybrid_Score": float(1 - (probs[primary] - probs[secondary])),
            })

    df = pd.DataFrame(hybrids)
    if not df.empty:
        df = df.sort_values("Hybrid_Score", ascending=False)
    return df
