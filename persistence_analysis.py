"""
Análise de persistência dos arquétipos entre temporadas (2024-25 -> 2025-26).

Pergunta: os mesmos arquétipos reaparecem na temporada seguinte e os jogadores
permanecem no mesmo arquétipo? Dialoga com Edwards-Thro (2021), que mediu
consistência de cluster entre temporadas (62,95%).

Procedimento:
  1. Clusteriza cada temporada de forma independente com K-Means k=3
     (StandardScaler ajustado por temporada, n_init=50, random_state=42).
  2. Ancora os clusters de cada temporada em identidades canônicas de arquétipo
     pelo perfil estatístico (Estrelas = maior PTS, Big Men = maior REB,
     Rotação = o restante), de modo que a numeração do K-Means (arbitrária) não
     contamine a comparação.
  3. Para os jogadores presentes nas duas temporadas, mede quantos mantiveram
     o arquétipo, quanto coincidiria por acaso (dados os tamanhos dos grupos)
     e reporta a matriz de transição e o ARI.

Uso:
    python persistence_analysis.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from config import CLUSTERING_FEATURES, KMEANS_CONFIG, SEASON, REPLICATION_SEASON
from src.data_collection import load_data

K = 3
# Identidades canônicas de arquétipo
BIG_MEN, ROTACAO, ESTRELAS = 0, 1, 2
ARCHETYPE_NAMES = {
    BIG_MEN: "Big Men/Defensivos",
    ROTACAO: "Rotação",
    ESTRELAS: "Estrelas/Criadores",
}
I_PTS = CLUSTERING_FEATURES.index("PTS")
I_REB = CLUSTERING_FEATURES.index("REB")


def cluster_season_canonical(df: pd.DataFrame):
    """
    Clusteriza uma temporada com K-Means k=3 e remapeia os rótulos para as
    identidades canônicas de arquétipo pelo perfil (PTS/REB na escala original).
    Retorna (labels_canonicos, perfis_canonicos[K x n_features]).
    """
    X = df[CLUSTERING_FEATURES].to_numpy()
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(
        n_clusters=K,
        n_init=KMEANS_CONFIG["n_init"],
        max_iter=KMEANS_CONFIG["max_iter"],
        random_state=KMEANS_CONFIG["random_state"],
    )
    raw = km.fit_predict(Xs)
    prof_raw = np.vstack([X[raw == c].mean(axis=0) for c in range(K)])

    # Atribuição canônica por assinatura de perfil
    estrelas = int(np.argmax(prof_raw[:, I_PTS]))           # maior pontuação
    restantes = [c for c in range(K) if c != estrelas]
    big = max(restantes, key=lambda c: prof_raw[c, I_REB])  # maior rebote
    rotacao = [c for c in restantes if c != big][0]
    raw_to_canon = {big: BIG_MEN, rotacao: ROTACAO, estrelas: ESTRELAS}

    labels = np.array([raw_to_canon[c] for c in raw])
    prof = np.vstack([X[labels == c].mean(axis=0) for c in range(K)])
    return labels, prof


def run_persistence(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Persistência dos arquétipos entre duas temporadas (a -> b)."""
    labels_a, prof_a = cluster_season_canonical(a)
    labels_b, prof_b = cluster_season_canonical(b)
    a = a.assign(arch=labels_a)
    b = b.assign(arch=labels_b)

    common = pd.merge(
        a[["PLAYER_ID", "PLAYER_NAME", "arch"]],
        b[["PLAYER_ID", "arch"]],
        on="PLAYER_ID", suffixes=("_a", "_b"),
    )
    n_common = len(common)
    same = int((common["arch_a"] == common["arch_b"]).sum())

    # Concordância esperada por acaso: soma dos produtos das proporções de
    # cada arquétipo nas duas temporadas (entre os jogadores em comum)
    share_a = common["arch_a"].value_counts(normalize=True)
    share_b = common["arch_b"].value_counts(normalize=True)
    chance = float(sum(share_a.get(c, 0) * share_b.get(c, 0) for c in range(K)))

    transitions = pd.crosstab(
        common["arch_a"].map(ARCHETYPE_NAMES),
        common["arch_b"].map(ARCHETYPE_NAMES),
    )
    return {
        "n_a": len(a),
        "n_b": len(b),
        "n_comum": n_common,
        "mantiveram": same,
        "pct_mantiveram": same / n_common,
        "pct_por_acaso": chance,
        "ari": float(adjusted_rand_score(common["arch_a"], common["arch_b"])),
        "tamanhos_a": {ARCHETYPE_NAMES[c]: int((labels_a == c).sum()) for c in range(K)},
        "tamanhos_b": {ARCHETYPE_NAMES[c]: int((labels_b == c).sum()) for c in range(K)},
        "perfis_a": {ARCHETYPE_NAMES[c]: dict(zip(CLUSTERING_FEATURES, prof_a[c].round(4).tolist())) for c in range(K)},
        "perfis_b": {ARCHETYPE_NAMES[c]: dict(zip(CLUSTERING_FEATURES, prof_b[c].round(4).tolist())) for c in range(K)},
        "transicoes": {de: {para: int(v) for para, v in row.items()} for de, row in transitions.iterrows()},
    }


def main():
    r = run_persistence(load_data(SEASON), load_data(REPLICATION_SEASON))
    print("=" * 70)
    print(f"PERSISTÊNCIA DE ARQUÉTIPOS: {SEASON} -> {REPLICATION_SEASON}")
    print("=" * 70)
    print(f"Jogadores nas duas temporadas: {r['n_comum']}")
    print(f"Mantiveram o arquétipo: {r['mantiveram']}/{r['n_comum']} = {r['pct_mantiveram']:.1%}")
    print(f"Esperado por acaso: {r['pct_por_acaso']:.1%}")
    print(f"ARI entre as duas atribuições: {r['ari']:.3f}")
    print("\nMatriz de transição (linha = primeira temporada, coluna = segunda):")
    print(pd.DataFrame(r["transicoes"]).T.to_string())


if __name__ == "__main__":
    main()
