"""
Análise de persistência dos arquétipos entre temporadas (2024-25 -> 2025-26).

Pergunta: os mesmos arquétipos reaparecem na temporada seguinte e os jogadores
permanecem no mesmo arquétipo? Responde a duas limitações declaradas na
conclusão (análise de uma única temporada e ausência de validação temporal) e
dialoga com Edwards-Thro (2021), que mediu consistência de cluster entre
temporadas (62,95%).

Procedimento:
  1. Clusteriza cada temporada de forma independente com K-Means k=3
     (StandardScaler ajustado por temporada, n_init=50, random_state=42).
  2. Ancora os clusters de cada temporada em identidades canônicas de arquétipo
     pelo perfil estatístico (Estrelas = maior PTS, Big Men = maior REB,
     Rotação = o restante), de modo que a numeração do K-Means (arbitrária) não
     contamine a comparação.
  3. Para os jogadores presentes nas duas temporadas, mede quantos mantiveram
     o arquétipo e reporta a matriz de transição e o ARI.
  4. (Bônus) Mostra os arquétipos dos elencos campeões: OKC (2024-25) e
     New York Knicks (2025-26).

Não altera o pipeline principal nem a temporada 2024-25 (oficial da monografia).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from config import ALL_FEATURES, KMEANS_CONFIG

K = 3
# Identidades canônicas de arquétipo
BIG_MEN, ROTACAO, ESTRELAS = 0, 1, 2
ARCHETYPE_NAMES = {
    BIG_MEN: "Big Men/Defensivos",
    ROTACAO: "Rotação",
    ESTRELAS: "Estrelas/Criadores",
}
I_PTS = ALL_FEATURES.index("PTS")
I_REB = ALL_FEATURES.index("REB")


def cluster_season_canonical(df: pd.DataFrame):
    """
    Clusteriza uma temporada com K-Means k=3 e remapeia os rótulos para as
    identidades canônicas de arquétipo pelo perfil (PTS/REB na escala original).
    Retorna (labels_canonicos, perfis_canonicos[K x n_features]).
    """
    X = df[ALL_FEATURES].to_numpy()
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


def main():
    a = pd.read_csv("data/nba_players_2024_25.csv")
    b = pd.read_csv("data/nba_players_2025_26.csv")

    print("=" * 70)
    print("PERSISTÊNCIA DE ARQUÉTIPOS: 2024-25  ->  2025-26")
    print("=" * 70)
    print(f"2024-25: {len(a)} jogadores | 2025-26: {len(b)} jogadores")

    labels_a, prof_a = cluster_season_canonical(a)
    labels_b, prof_b = cluster_season_canonical(b)
    a = a.copy(); a["arch"] = labels_a
    b = b.copy(); b["arch"] = labels_b

    # Perfis lado a lado (confirma que os 3 arquétipos reaparecem)
    print("\n--- Perfis dos arquétipos (médias na escala original) ---")
    for c in range(K):
        na = int((labels_a == c).sum()); nb = int((labels_b == c).sum())
        print(f"\nArquétipo {c} ({ARCHETYPE_NAMES[c]}):  2024-25 n={na} | 2025-26 n={nb}")
        print("  feat      2024-25   2025-26")
        for j, f in enumerate(ALL_FEATURES):
            print(f"  {f:7s} {prof_a[c, j]:8.2f}  {prof_b[c, j]:8.2f}")

    # Persistência nos jogadores presentes nas duas temporadas
    common = pd.merge(
        a[["PLAYER_ID", "PLAYER_NAME", "arch"]],
        b[["PLAYER_ID", "arch"]],
        on="PLAYER_ID", suffixes=("_2425", "_2526"),
    )
    n_common = len(common)
    same = int((common["arch_2425"] == common["arch_2526"]).sum())
    pct = same / n_common * 100
    ari = adjusted_rand_score(common["arch_2425"], common["arch_2526"])

    print("\n" + "=" * 70)
    print("PERSISTÊNCIA (jogadores presentes nas duas temporadas)")
    print("=" * 70)
    print(f"Jogadores em ambas: {n_common}")
    print(f"Mantiveram o arquétipo: {same}/{n_common} = {pct:.1f}%")
    print(f"ARI entre as duas atribuições (jogadores comuns): {ari:.3f}")
    print("(Referência: Edwards-Thro 2021 reportou 62,95% de consistência.)")

    print("\n--- Matriz de transição (linha = 2024-25, coluna = 2025-26) ---")
    ct = pd.crosstab(
        common["arch_2425"].map(ARCHETYPE_NAMES),
        common["arch_2526"].map(ARCHETYPE_NAMES),
    )
    print(ct.to_string())

    changed = common[common["arch_2425"] != common["arch_2526"]].copy()
    changed["de"] = changed["arch_2425"].map(ARCHETYPE_NAMES)
    changed["para"] = changed["arch_2526"].map(ARCHETYPE_NAMES)
    print(f"\nMudaram de arquétipo: {len(changed)} jogadores")
    print(changed[["PLAYER_NAME", "de", "para"]].head(15).to_string(index=False))

    # BÔNUS: arquétipos dos campeões
    print("\n" + "=" * 70)
    print("BÔNUS: ARQUÉTIPOS DOS CAMPEÕES")
    print("=" * 70)
    for season_df, team, season in [(a, "OKC", "2024-25"), (b, "NYK", "2025-26")]:
        roster = season_df[season_df["TEAM_ABBREVIATION"] == team].sort_values(
            "PTS", ascending=False)
        print(f"\n{team} (campeão {season}) — {len(roster)} jogadores no recorte (>=500 min):")
        for _, row in roster.iterrows():
            print(f"  {row['PLAYER_NAME']:26s} {ARCHETYPE_NAMES[int(row['arch'])]:20s} "
                  f"PTS={row['PTS']:5.1f} REB={row['REB']:4.1f} AST={row['AST']:4.1f}")


if __name__ == "__main__":
    main()
