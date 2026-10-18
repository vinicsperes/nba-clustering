"""
NBA Player Archetypes - Pipeline Principal
Análise Comparativa de K-Means, GMM e Clustering Hierárquico

Pergunta de Pesquisa:
"Qual algoritmo de clustering é mais adequado para identificar agrupamentos
de jogadores da NBA com base em estatísticas de desempenho?"

H0: Os três algoritmos produzem agrupamentos com qualidade equivalente.
H1: Existe diferença significativa na qualidade dos agrupamentos.

Cada rodada (temporada + filtro) grava em results/<rodada>/:
  numeros.json      todos os números citados na monografia
  relatorio.md      os mesmos números em tabelas, para leitura
  atribuicoes.csv   cluster de cada jogador em cada algoritmo
  bootstrap_pareado.csv
  figuras/          as oito figuras da monografia

Uso:
    python main.py                                   # todas as rodadas
    python main.py --season 2024-25                  # só uma rodada
    python main.py --season 2024-25 --filtro 15mpg   # variante do recorte
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config import (
    SEASON,
    REPLICATION_SEASON,
    DEFAULT_FILTER,
    HYBRID_THRESHOLD,
    RESULTS_DIR,
    results_path,
)
from src.data_collection import load_data
from src.preprocessing import prepare_data, get_position_labels
from src.clustering import (
    run_clustering,
    get_cluster_profiles,
    get_cluster_examples,
    analyze_hybrid_players,
)
from src.validation import compare_algorithms, crosstab_clusters_positions
from src.visualization import create_figures
from src.report import write_report
from friedman_analysis import run_friedman
from sensitivity_features import run_sensitivity
from persistence_analysis import run_persistence

warnings.filterwarnings("ignore")

ALGOS = ["kmeans", "gmm", "hierarchical"]

# Rodadas geradas por `python main.py`
RUNS = [
    (SEASON, DEFAULT_FILTER),             # análise principal
    (SEASON, "15mpg"),                    # sensibilidade ao recorte
    (REPLICATION_SEASON, DEFAULT_FILTER), # replicação
]


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Tipo não serializável: {type(o)}")


def _table(df: pd.DataFrame) -> dict:
    """Tabela como {linha: {coluna: valor}}, com chaves em texto (JSON)."""
    return {str(i): {str(c): v for c, v in row.items()} for i, row in df.to_dict("index").items()}


def _save_json(data: dict, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def run(season: str, filtro: str = DEFAULT_FILTER) -> dict:
    """Executa o pipeline completo de uma rodada e grava os resultados."""
    out_dir = results_path(season, filtro)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"RODADA: {season} ({filtro}) -> {out_dir}")
    print("=" * 70)

    df = load_data(season, filtro)
    X, preprocessor = prepare_data(df)
    features = preprocessor.features_used
    positions = get_position_labels(df)
    names = df["PLAYER_NAME"].tolist()

    # Clustering: cada algoritmo escolhe o k pelo seu critério próprio
    clustering = run_clustering(X)
    results = {algo: clustering[algo] for algo in ALGOS}
    ks = {algo: results[algo].k for algo in ALGOS}

    # Métricas internas, estabilidade e concordância com as posições
    print("\nComparação dos algoritmos...")
    comparison = compare_algorithms(X, results, positions)

    # Teste formal (bootstrap pareado + Friedman + Wilcoxon)
    print("\nTeste de Friedman...")
    ari_df, fried, wilc = run_friedman(X, ks)

    # Sensibilidade às features de rebote
    print("\nSensibilidade às features...")
    sensitivity = run_sensitivity(df)

    gmm = results["gmm"]
    hybrids = analyze_hybrid_players(gmm.probabilities, names, threshold=HYBRID_THRESHOLD)
    create_figures(X, clustering, features, out_dir / "figuras")

    bic = clustering["gmm_model"].get_bic_data()
    describe = df[features].describe()

    numeros = {
        "rodada": {"temporada": season, "filtro": filtro},
        "amostra": {
            "n_jogadores": len(df),
            "n_colunas": df.shape[1],
            "posicoes": (pd.Series(positions).value_counts().to_dict()
                         if positions is not None else None),
            "fg3_pct_zero": int((df["FG3_PCT"] == 0).sum()),
            "sem_tentativa_de_3": int((df["FG3A"] == 0).sum()) if "FG3A" in df else None,
        },
        "descritivas": {
            f: {
                "media": describe.loc["mean", f],
                "desvio": describe.loc["std", f],
                "minimo": describe.loc["min", f],
                "mediana": describe.loc["50%", f],
                "maximo": describe.loc["max", f],
            }
            for f in features
        },
        "correlacoes": pd.DataFrame(X, columns=features).corr().to_dict(),
        "pca_variancia": PCA(n_components=2, random_state=42).fit(X).explained_variance_ratio_,
        "selecao_k": {
            "kmeans": clustering["kmeans_model"].get_elbow_data().to_dict("records"),
            "gmm": bic.to_dict("records"),
            "hierarchical": clustering["hier_model"].get_silhouette_data().to_dict("records"),
            "gmm_k_por_silhouette": int(bic.loc[bic["silhouette"].idxmax(), "k"]),
        },
        "tamanhos": {algo: np.bincount(results[algo].labels).tolist() for algo in ALGOS},
        "comparacao": {
            algo: {
                "k": row["k"],
                "silhouette": row["Silhouette Score"],
                "davies_bouldin": row["Davies-Bouldin Index"],
                "calinski_harabasz": row["Calinski-Harabasz Index"],
                "ari_bootstrap": row["Bootstrap Mean ARI"],
                "ari_sementes": row["Random Seed Mean ARI"],
                "silhouette_por_cluster": row["Silhouette per Cluster"],
                "ari_posicoes": row.get("ARI vs Positions"),
                "nmi_posicoes": row.get("NMI vs Positions"),
            }
            for algo, (_, row) in zip(ALGOS, comparison.iterrows())
        },
        "friedman": {
            "chi2": fried["statistic"],
            "p": fried["p_value"],
            "ari_pareado": fried["means"],
            "rejeita_h0": fried["reject_h0"],
            "wilcoxon": wilc.to_dict("records"),
        },
        "sensibilidade_features": sensitivity,
        "perfis": {
            algo: get_cluster_profiles(X, results[algo].labels, features, df).to_dict("records")
            for algo in ALGOS
        },
        "exemplos": {
            algo: get_cluster_examples(X, results[algo].labels, names, n_examples=5)
            for algo in ALGOS
        },
        "posicoes_por_cluster": (
            {algo: _table(crosstab_clusters_positions(results[algo].labels, positions))
             for algo in ALGOS}
            if positions is not None else None
        ),
        "hibridos": hybrids.to_dict("records"),
    }

    _save_json(numeros, out_dir / "numeros.json")
    write_report(numeros, out_dir / "relatorio.md")
    ari_df.to_csv(out_dir / "bootstrap_pareado.csv", index=False)

    assignments = df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]].copy()
    if positions is not None:
        assignments["POSITION"] = positions
    for algo in ALGOS:
        assignments[f"{algo.upper()}_CLUSTER"] = results[algo].labels
    for i in range(gmm.probabilities.shape[1]):
        assignments[f"GMM_PROB_{i}"] = gmm.probabilities[:, i]
    assignments.to_csv(out_dir / "atribuicoes.csv", index=False)

    print(f"\nk: {ks}")
    for algo, c in numeros["comparacao"].items():
        print(f"  {algo:13s} Sil={c['silhouette']:.3f}  DB={c['davies_bouldin']:.3f}  "
              f"CH={c['calinski_harabasz']:.2f}  ARI bootstrap={c['ari_bootstrap']:.3f}")
    print(f"  Friedman chi2={fried['statistic']:.2f}  p={fried['p_value']:.3e}")
    print(f"Resultados em {out_dir}/\n")
    return numeros


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo do TCC.")
    parser.add_argument("--season", help="Roda só esta temporada (padrão: todas as rodadas).")
    parser.add_argument("--filtro", default=DEFAULT_FILTER)
    args = parser.parse_args()

    if args.season:
        run(args.season, args.filtro)
        return

    for season, filtro in RUNS:
        run(season, filtro)

    persistence = run_persistence(load_data(SEASON), load_data(REPLICATION_SEASON))
    _save_json(persistence, RESULTS_DIR / "persistencia.json")
    print(f"Persistência: {persistence['pct_mantiveram']:.1%} mantiveram o arquétipo "
          f"(por acaso: {persistence['pct_por_acaso']:.1%}; ARI {persistence['ari']:.3f})")
    print(f"Resultados em {RESULTS_DIR / 'persistencia.json'}")


if __name__ == "__main__":
    main()
