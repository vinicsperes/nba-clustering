"""
Módulo de pré-processamento de dados para clustering.

Responsabilidades:
- Seleção das features de clustering disponíveis no DataFrame
- Normalização via StandardScaler
- PCA opcional para redução de dimensionalidade
- Extração de labels de posição para validação externa
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import List, Optional, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import CLUSTERING_FEATURES, POSITION_GROUPS


class DataPreprocessor:
    """
    Preprocessador de dados para clustering.

    Seleciona as features disponíveis, normaliza via StandardScaler
    e aplica PCA quando solicitado.
    """

    def __init__(self, features: Optional[List[str]] = None):
        self.requested_features = features or CLUSTERING_FEATURES
        self.features_used: List[str] = []
        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None
        self.is_fitted: bool = False

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [f for f in self.requested_features if f in df.columns]

        if len(available) < 3:
            raise ValueError(
                f"Features insuficientes para clustering: {len(available)} disponíveis. "
                f"Necessário mínimo de 3."
            )

        self.features_used = available
        return df[available].copy()

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        X = self._select_features(df)
        self.scaler.fit(X)
        self.is_fitted = True
        print(f"Preprocessador ajustado: {len(X)} amostras, {len(self.features_used)} features")
        print(f"  Features: {self.features_used}")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Preprocessador não ajustado. Use fit() primeiro.")
        return self.scaler.transform(df[self.features_used])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    def apply_pca(
        self,
        X: np.ndarray,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.85,
    ) -> np.ndarray:
        if n_components is None:
            cumsum = np.cumsum(PCA().fit(X).explained_variance_ratio_)
            n_components = max(2, int(np.argmax(cumsum >= variance_threshold) + 1))

        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)

        print(f"\nPCA: {n_components} componentes, "
              f"{self.pca.explained_variance_ratio_.sum():.1%} de variância explicada")
        return X_pca


def prepare_data(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    apply_pca: bool = False,
    pca_variance: float = 0.85,
) -> Tuple[np.ndarray, pd.DataFrame, DataPreprocessor]:
    """
    Pipeline completo de preparação de dados.

    Returns
    -------
    X : np.ndarray
        Matriz de features normalizada, pronta para clustering.
    df_meta : pd.DataFrame
        Metadados dos jogadores (nome, time, etc.).
    preprocessor : DataPreprocessor
        Objeto ajustado para uso posterior.
    """
    print("=" * 50)
    print("PRÉ-PROCESSAMENTO")
    print("=" * 50)

    preprocessor = DataPreprocessor(features=features)
    X = preprocessor.fit_transform(df)
    print(f"  Shape após normalização: {X.shape}")

    if apply_pca:
        X = preprocessor.apply_pca(X, variance_threshold=pca_variance)

    meta_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE", "GP", "MIN"]
    df_meta = df[[c for c in meta_cols if c in df.columns]].copy()

    print("=" * 50)
    return X, df_meta, preprocessor


def get_position_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Extrai labels de posição simplificadas (Guard / Forward / Center)
    para uso na validação externa.
    """
    def map_position(pos) -> str:
        if pd.isna(pos):
            return "Unknown"
        pos = str(pos).upper().strip()
        for group, positions in POSITION_GROUPS.items():
            if any(p in pos for p in positions):
                return group
        return "Unknown"

    if "POSITION_SIMPLE" in df.columns:
        labels = df["POSITION_SIMPLE"].fillna("Unknown").values
        n_known = (labels != "Unknown").sum()
        print(f"Posições extraídas de 'POSITION_SIMPLE': {n_known}/{len(labels)} conhecidas")
        return labels

    for col in ["POSITION", "POS", "PLAYER_POSITION"]:
        if col in df.columns:
            labels = df[col].apply(map_position).values
            n_known = (labels != "Unknown").sum()
            print(f"Posições extraídas de '{col}': {n_known}/{len(labels)} conhecidas")
            return labels

    print("AVISO: coluna de posição não encontrada.")
    return np.array(["Unknown"] * len(df))
