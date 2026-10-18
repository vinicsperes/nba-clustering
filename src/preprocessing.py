"""
Módulo de pré-processamento de dados para clustering.

Responsabilidades:
- Seleção das features de clustering disponíveis no DataFrame
- Normalização via StandardScaler
- Extração de labels de posição para validação externa
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import CLUSTERING_FEATURES


class DataPreprocessor:
    """
    Preprocessador de dados para clustering.

    Seleciona as features disponíveis e normaliza via StandardScaler.
    """

    def __init__(self, features: Optional[List[str]] = None):
        self.requested_features = features or CLUSTERING_FEATURES
        self.features_used: List[str] = []
        self.scaler = StandardScaler()
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
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Preprocessador não ajustado. Use fit() primeiro.")
        return self.scaler.transform(df[self.features_used])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)


def prepare_data(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> Tuple[np.ndarray, DataPreprocessor]:
    """
    Seleciona as features e normaliza.

    Returns
    -------
    X : np.ndarray
        Matriz de features normalizada, pronta para clustering.
    preprocessor : DataPreprocessor
        Objeto ajustado para uso posterior.
    """
    preprocessor = DataPreprocessor(features=features)
    X = preprocessor.fit_transform(df)
    return X, preprocessor


def get_position_labels(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Labels de posição simplificadas (Guard / Forward / Center) para a
    validação externa, ou None se o CSV não tiver posições.
    """
    if "POSITION_SIMPLE" not in df.columns:
        return None
    return df["POSITION_SIMPLE"].fillna("Unknown").values
