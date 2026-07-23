"""
NBA Player Archetypes - Pipeline Principal
Análise Comparativa de K-Means, GMM e Clustering Hierárquico

Pergunta de Pesquisa:
"Qual algoritmo de clustering é mais adequado para identificar agrupamentos
de jogadores da NBA com base em estatísticas de desempenho?"

Hipótese:
H₀: Os três algoritmos produzem agrupamentos com qualidade equivalente.
H₁: Existe diferença significativa na qualidade dos agrupamentos entre os algoritmos.

Pipeline:
1. Coleta de dados (NBA API) + auto-patch posições
2. Pré-processamento (seleção de features + normalização)
3. Clustering (K-Means + GMM + Hierárquico)
4. Validação (métricas internas + externas por algoritmo)
5. Comparação de algoritmos (tabela comparativa + estabilidade)
6. Clusters vs Posições (análise secundária)
7. Perfis e arquétipos
8. Jogadores híbridos (GMM)
9. Visualizações
10. Salvar resultados
"""

import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from config import DATA_DIR, RESULTS_DIR, FIGURES_DIR, SEASON
from src.data_collection import collect_all_data, load_data, patch_positions
from src.preprocessing import prepare_data, get_position_labels
from src.clustering import (
    run_clustering,
    get_cluster_profiles,
    get_cluster_examples,
    analyze_hybrid_players,
)
from src.validation import (
    validate_clustering,
    generate_validation_report,
    compare_algorithms,
    crosstab_clusters_positions,
)
from src.visualization import create_all_visualizations

warnings.filterwarnings("ignore")


def main(
    collect_new_data: bool = False,
    apply_pca: bool = False,
    save_results: bool = True,
):
    """
    Pipeline principal de análise.

    Parameters
    ----------
    collect_new_data : bool
        Se True, coleta dados frescos da NBA API.
        Se False, usa dados já salvos.
    apply_pca : bool
        Se True, aplica PCA antes do clustering.
    save_results : bool
        Se True, salva resultados em CSV.

    Returns
    -------
    dict
        Resultados completos da análise
    """
    print("=" * 70)
    print("TCC: ANÁLISE COMPARATIVA DE K-MEANS, GMM E CLUSTERING HIERÁRQUICO")
    print("Identificação de Arquétipos de Jogadores NBA")
    print("=" * 70)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Temporada: {SEASON}")
    print(f"\nH₀: Algoritmos produzem agrupamentos com qualidade equivalente")
    print(f"H₁: Existe diferença significativa na qualidade dos agrupamentos")

    # =========================================================================
    # ETAPA 1: COLETA DE DADOS
    # =========================================================================
    print("\n" + "=" * 70)
    print("ETAPA 1: COLETA DE DADOS")
    print("=" * 70)

    if collect_new_data:
        df = collect_all_data(season=SEASON)
    else:
        try:
            df = load_data(season=SEASON)
        except FileNotFoundError:
            print("Dados não encontrados. Coletando...")
            df = collect_all_data(season=SEASON)

    # Auto-patch posições se necessário
    needs_patch = "POSITION" not in df.columns
    if not needs_patch and "POSITION" in df.columns:
        n_unknown = (df["POSITION"] == "Unknown").sum()
        if n_unknown > len(df) * 0.5:
            print(f"\n{n_unknown}/{len(df)} posições são 'Unknown'. Re-buscando...")
            needs_patch = True

    if needs_patch:
        print("\nBuscando posições dos jogadores...")
        try:
            df = patch_positions(season=SEASON)
        except Exception as e:
            print(f"  Não foi possível buscar posições: {e}")

    print(f"\nDataset: {len(df)} jogadores, {len(df.columns)} colunas")

    # =========================================================================
    # ETAPA 2: PRÉ-PROCESSAMENTO
    # =========================================================================
    print("\n" + "=" * 70)
    print("ETAPA 2: PRÉ-PROCESSAMENTO")
    print("=" * 70)

    X, df_meta, preprocessor = prepare_data(df, apply_pca=apply_pca)

    # Extrair posições para validação
    positions = get_position_labels(df)

    # Mostrar distribuição de posições
    print("\nDistribuição de Posições Tradicionais:")
    unique, counts = np.unique(positions, return_counts=True)
    for pos, count in zip(unique, counts):
        pct = count / len(positions) * 100
        print(f"  {pos}: {count} ({pct:.1f}%)")

    # =========================================================================
    # ETAPA 3: CLUSTERING (K-Means + GMM + Hierárquico)
    # =========================================================================
    print("\n" + "=" * 70)
    print("ETAPA 3: CLUSTERING")
    print("=" * 70)

    clustering_results = run_clustering(X)

    kmeans_result = clustering_results["kmeans"]
    gmm_result = clustering_results["gmm"]
    hier_result = clustering_results["hierarchical"]

    print(f"\nResultados:")
    print(f"  K-Means:      k={kmeans_result.k}, Silhouette={kmeans_result.silhouette:.3f}")
    print(f"  GMM:           k={gmm_result.k}, Silhouette={gmm_result.silhouette:.3f}")
    print(f"  Hierárquico:   k={hier_result.k}, Silhouette={hier_result.silhouette:.3f}")

    # =========================================================================
    # ETAPA 4: VALIDAÇÃO CIENTÍFICA (por algoritmo)
    # =========================================================================
    print("\n" + "=" * 70)
    print("ETAPA 4: VALIDAÇÃO CIENTÍFICA")
    print("=" * 70)

    validations = {}
    for algo_name, result in [("K-Means", kmeans_result), ("GMM", gmm_result), ("Hierarchical", hier_result)]:
        validations[algo_name] = validate_clustering(
            X, result.labels, positions, algorithm_name=algo_name
        )

    # =========================================================================
    # ETAPA 5: COMPARAÇÃO DE ALGORITMOS
    # =========================================================================
    print("\n" + "=" * 70)
    print("ETAPA 5: COMPARAÇÃO DE ALGORITMOS")
    print("=" * 70)

    comparison_df = compare_algorithms(
        X,
        {"kmeans": kmeans_result, "gmm": gmm_result, "hierarchical": hier_result},
        positions,
    )

    # =========================================================================
    # ETAPA 6: CLUSTERS VS POSIÇÕES REAIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("ETAPA 6: CLUSTERS VS POSIÇÕES REAIS")
    print("=" * 70)

    crosstabs = {}
    for algo_name, result in [("K-Means", kmeans_result), ("GMM", gmm_result), ("Hierarchical", hier_result)]:
        crosstabs[algo_name] = crosstab_clusters_positions(
            result.labels, positions, algorithm_name=algo_name
        )

    # =========================================================================
    # ETAPA 7: PERFIS E ARQUÉTIPOS
    # =========================================================================
    print("\n" + "=" * 70)
    print("ETAPA 7: PERFIS DOS CLUSTERS")
    print("=" * 70)

    player_names = df["PLAYER_NAME"].tolist()
    all_profiles = {}
    all_examples = {}

    for algo_key, result in [("kmeans", kmeans_result), ("gmm", gmm_result), ("hierarchical", hier_result)]:
        print(f"\n--- {result.algorithm} (k={result.k}) ---")

        profiles = get_cluster_profiles(
            X, result.labels, preprocessor.features_used, df
        )
        all_profiles[algo_key] = profiles
        print(profiles.to_string(index=False))

        examples = get_cluster_examples(result.labels, player_names, n_examples=5)
        all_examples[algo_key] = examples

        print(f"\nExemplos de Jogadores:")
        for cluster, players in examples.items():
            print(f"  Cluster {cluster}: {', '.join(players)}")

    # =========================================================================
    # ETAPA 8: JOGADORES HÍBRIDOS (GMM)
    # =========================================================================
    hybrids = pd.DataFrame()
    if gmm_result.probabilities is not None:
        print("\n" + "=" * 70)
        print("ETAPA 8: JOGADORES HÍBRIDOS (GMM - Soft Clustering)")
        print("=" * 70)

        hybrids = analyze_hybrid_players(
            gmm_result.probabilities,
            player_names,
            threshold=0.25
        )

        if not hybrids.empty:
            print(f"\n{len(hybrids)} jogadores híbridos identificados:")
            print(hybrids.head(15).to_string(index=False))
        else:
            print("Nenhum jogador híbrido significativo encontrado.")

    # =========================================================================
    # ETAPA 9: VISUALIZAÇÕES
    # =========================================================================
    print("\n" + "=" * 70)
    print("ETAPA 9: VISUALIZAÇÕES")
    print("=" * 70)

    # Preparar dados para visualizações
    clustering_labels = {
        "K-Means": kmeans_result.labels,
        "GMM": gmm_result.labels,
        "Hierarchical": hier_result.labels,
    }

    # Dados do K-Means para elbow curve
    kmeans_model = clustering_results["kmeans_model"]
    elbow_data = kmeans_model.get_elbow_data()

    # Dados do GMM para a curva do BIC
    gmm_model = clustering_results["gmm_model"]
    bic_data = gmm_model.get_bic_data()

    analysis_results = {
        "kmeans": {
            "k_values": elbow_data["k"].tolist(),
            "inertias": elbow_data["inertia"].tolist(),
            "silhouettes": elbow_data["silhouette"].tolist(),
            "optimal_k": kmeans_result.k,
        },
        "gmm": {
            "k_values": bic_data["k"].tolist(),
            "bics": bic_data["bic"].tolist(),
            "silhouettes": bic_data["silhouette"].tolist(),
            "optimal_k": gmm_result.k,
        },
        "hierarchical": {
            "linkage_matrix": hier_result.linkage_matrix,
            "optimal_k": hier_result.k,
        },
        "comparison": comparison_df,
    }

    # Adicionar stability_df para visualização
    stability_cols = ["Algorithm", "Random Seed Mean ARI", "Bootstrap Mean ARI", "Mean Point Stability"]
    available_stab = [c for c in stability_cols if c in comparison_df.columns]
    if len(available_stab) == len(stability_cols):
        analysis_results["stability"] = comparison_df[stability_cols]

    try:
        create_all_visualizations(
            X,
            clustering_labels,
            analysis_results,
            preprocessor.features_used,
            player_names=player_names,
            output_dir=FIGURES_DIR,
        )
    except Exception as e:
        print(f"  AVISO: Erro ao criar visualizações: {e}")

    # =========================================================================
    # ETAPA 10: SALVAR RESULTADOS
    # =========================================================================
    if save_results:
        print("\n" + "=" * 70)
        print("ETAPA 10: SALVAR RESULTADOS")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Salvar atribuições de jogadores
        assignments = df[["PLAYER_NAME", "TEAM_ABBREVIATION"]].copy()
        assignments["POSITION"] = positions
        assignments["KMEANS_CLUSTER"] = kmeans_result.labels
        assignments["GMM_CLUSTER"] = gmm_result.labels
        assignments["HIER_CLUSTER"] = hier_result.labels

        # Adicionar probabilidades do GMM
        if gmm_result.probabilities is not None:
            for i in range(gmm_result.probabilities.shape[1]):
                assignments[f"GMM_PROB_{i}"] = gmm_result.probabilities[:, i]

        assign_path = RESULTS_DIR / f"player_assignments_{timestamp}.csv"
        assignments.to_csv(assign_path, index=False)
        print(f"  Atribuições salvas: {assign_path}")

        # Salvar perfis dos clusters (K-Means como principal)
        profiles_path = RESULTS_DIR / f"cluster_profiles_{timestamp}.csv"
        all_profiles["kmeans"].to_csv(profiles_path, index=False)
        print(f"  Perfis salvos: {profiles_path}")

        # Salvar comparação de algoritmos
        comparison_path = RESULTS_DIR / f"algorithm_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"  Comparação salva: {comparison_path}")

        # Salvar relatório de validação (K-Means principal)
        report = generate_validation_report(
            validations["K-Means"], all_profiles["kmeans"],
            algorithm_name="K-Means"
        )
        report_path = RESULTS_DIR / f"validation_report_{timestamp}.txt"
        report_path.write_text(report)
        print(f"  Relatório salvo: {report_path}")

        # Salvar jogadores híbridos
        if not hybrids.empty:
            hybrids_path = RESULTS_DIR / f"hybrid_players_{timestamp}.csv"
            hybrids.to_csv(hybrids_path, index=False)
            print(f"  Híbridos salvos: {hybrids_path}")

    # =========================================================================
    # CONCLUSÃO
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONCLUSÃO - TESTE DE HIPÓTESE")
    print("=" * 70)

    # Ranking por cada métrica
    print("\n--- Ranking por Métrica ---")
    best_sil = comparison_df.loc[comparison_df["Silhouette Score"].idxmax()]
    best_db = comparison_df.loc[comparison_df["Davies-Bouldin Index"].idxmin()]
    best_ch = comparison_df.loc[comparison_df["Calinski-Harabasz Index"].idxmax()]

    print(f"  Silhouette Score (maior=melhor):   {best_sil['Algorithm']} ({best_sil['Silhouette Score']:.4f})")
    print(f"  Davies-Bouldin (menor=melhor):     {best_db['Algorithm']} ({best_db['Davies-Bouldin Index']:.4f})")
    print(f"  Calinski-Harabasz (maior=melhor):  {best_ch['Algorithm']} ({best_ch['Calinski-Harabasz Index']:.2f})")

    if "Bootstrap Mean ARI" in comparison_df.columns:
        best_boot = comparison_df.loc[comparison_df["Bootstrap Mean ARI"].idxmax()]
        print(f"  Estabilidade Bootstrap (maior=melhor): {best_boot['Algorithm']} ({best_boot['Bootstrap Mean ARI']:.4f})")

    # Verificar se o mesmo algoritmo vence em todas as métricas
    winners = {best_sil['Algorithm'], best_db['Algorithm'], best_ch['Algorithm']}

    print(f"\n--- Teste de Hipótese ---")
    print(f"H₀: Os três algoritmos produzem agrupamentos com qualidade equivalente.")
    print(f"H₁: Existe diferença significativa na qualidade dos agrupamentos.\n")

    # Calcular spread das métricas para avaliar se diferença é significativa
    sil_range = comparison_df["Silhouette Score"].max() - comparison_df["Silhouette Score"].min()
    sil_mean = comparison_df["Silhouette Score"].mean()

    if len(winners) == 1:
        winner = winners.pop()
        print(f"RESULTADO: H₀ REJEITADA")
        print(f"  {winner} supera os demais algoritmos em todas as métricas internas.")
        print(f"  Diferença no Silhouette Score: {sil_range:.4f} "
              f"({sil_range/sil_mean*100:.1f}% do valor médio)")
    else:
        if sil_range > 0.05:
            print(f"RESULTADO: H₀ REJEITADA")
            print(f"  Há diferença significativa entre os algoritmos.")
            print(f"  Spread no Silhouette Score: {sil_range:.4f}")
        else:
            print(f"RESULTADO: H₀ NÃO REJEITADA")
            print(f"  As diferenças entre os algoritmos são pequenas.")
            print(f"  Spread no Silhouette Score: {sil_range:.4f}")

    # Melhor algoritmo geral
    print(f"\nMELHOR ALGORITMO: {best_sil['Algorithm']} "
          f"(k={int(best_sil['k'])}, Silhouette={best_sil['Silhouette Score']:.4f})")

    # Resumo descritivo
    print(f"\n--- Resumo Descritivo (melhor algoritmo) ---")
    print(validations[best_sil['Algorithm']].evidence_summary)

    return {
        "df": df,
        "X": X,
        "preprocessor": preprocessor,
        "positions": positions,
        "kmeans": kmeans_result,
        "gmm": gmm_result,
        "hierarchical": hier_result,
        "validations": validations,
        "comparison": comparison_df,
        "profiles": all_profiles,
        "examples": all_examples,
        "hybrids": hybrids,
    }


if __name__ == "__main__":
    results = main(
        collect_new_data=False,
        apply_pca=False,
        save_results=True,
    )
