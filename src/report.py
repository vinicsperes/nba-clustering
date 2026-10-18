"""
Relatório de leitura (Markdown) a partir do numeros.json de uma rodada.

Os números aparecem com vírgula decimal, como na monografia.
"""

from pathlib import Path
from typing import Dict, List

ALGO_NAMES = {"kmeans": "K-Means", "gmm": "GMM (BIC)", "hierarchical": "Hierárquico",
              "gmm_silhouette": "GMM (Silhouette)"}

FIGURES = [
    ("correlation_matrix.png", "Matriz de correlação"),
    ("kmeans_elbow.png", "K-Means: cotovelo e Silhouette por k"),
    ("k-means_pca.png", "K-Means: projeção PCA"),
    ("k-means_silhouette.png", "K-Means: diagrama de silhueta"),
    ("gmm_bic.png", "GMM: BIC e Silhouette por k"),
    ("gmm_pca.png", "GMM: projeção PCA"),
    ("hierarchical_dendrogram.png", "Hierárquico: dendrograma"),
    ("hierarchical_pca.png", "Hierárquico: projeção PCA"),
]


def num(x, d: int = 3) -> str:
    """Número com vírgula decimal."""
    return f"{x:.{d}f}".replace(".", ",")


def sci(x) -> str:
    """p-valor em notação científica com vírgula decimal."""
    return f"{x:.2e}".replace(".", ",")


def table(header: List[str], rows: List[List]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write_report(n: Dict, path: Path) -> None:
    r, a = n["rodada"], n["amostra"]
    out = [f"# Resultados: temporada {r['temporada']}, filtro {r['filtro']}", ""]
    out.append("Gerado por `main.py` a partir de `numeros.json`. Não editar à mão.")

    out += ["", "## Amostra", ""]
    out.append(f"- Jogadores: {a['n_jogadores']}")
    if a["posicoes"]:
        total = sum(a["posicoes"].values())
        out.append("- Posições: " + ", ".join(
            f"{p} {c} ({num(100 * c / total, 1)}%)" for p, c in a["posicoes"].items()))
    out.append(f"- FG3_PCT = 0: {a['fg3_pct_zero']} jogadores"
               + (f" ({a['sem_tentativa_de_3']} sem nenhuma tentativa de três)"
                  if a["sem_tentativa_de_3"] is not None else ""))

    out += ["", "## Estatísticas descritivas", ""]
    out.append(table(
        ["Variável", "Média", "Desvio", "Mínimo", "Mediana", "Máximo"],
        [[f] + [num(v[k], 3 if f.endswith("PCT") else 2)
                for k in ["media", "desvio", "minimo", "mediana", "maximo"]]
         for f, v in n["descritivas"].items()],
    ))

    corr = n["correlacoes"]
    feats = list(corr)
    pairs = sorted(
        ((corr[f][g], f, g) for i, f in enumerate(feats) for g in feats[i + 1:]),
        key=lambda t: -abs(t[0]),
    )
    out += ["", "## Correlações mais fortes (|r| >= 0,5)", ""]
    out.append(table(["Par", "r"], [[f"{f} x {g}", num(c, 2)] for c, f, g in pairs if abs(c) >= 0.5]))
    pc = n["pca_variancia"]
    out.append(f"\nPCA (só para as figuras): PC1 {num(100 * pc[0], 1)}%, PC2 {num(100 * pc[1], 1)}%, "
               f"juntas {num(100 * sum(pc), 1)}%.")

    sk = n["selecao_k"]
    out += ["", "## Seleção de k", ""]
    out.append(table(
        ["k", "K-Means inércia", "K-Means Sil.", "GMM BIC", "GMM Sil.", "Hier. Sil."],
        [[km["k"], num(km["inertia"], 1), num(km["silhouette"]), num(g["bic"], 1),
          num(g["silhouette"]), num(h["silhouette"])]
         for km, g, h in zip(sk["kmeans"], sk["gmm"], sk["hierarchical"])],
    ))
    out.append(f"\nEscolhidos: K-Means k={n['comparacao']['kmeans']['k']}, "
               f"GMM k={n['comparacao']['gmm']['k']} (pelo Silhouette seria {sk['gmm_k_por_silhouette']}), "
               f"Hierárquico k={n['comparacao']['hierarchical']['k']}.")
    out.append("\nTamanhos: " + "; ".join(
        f"{ALGO_NAMES[algo]} {list(t)}" for algo, t in n["tamanhos"].items()))

    out += ["", "## Comparação dos algoritmos", ""]
    rows = []
    for algo, c in n["comparacao"].items():
        row = [ALGO_NAMES[algo], c["k"], num(c["silhouette"]), num(c["davies_bouldin"]),
               num(c["calinski_harabasz"], 2), num(c["ari_bootstrap"]), num(c["ari_sementes"])]
        if c["ari_posicoes"] is not None:
            row += [num(c["ari_posicoes"]), num(c["nmi_posicoes"])]
        rows.append(row)
    header = ["Algoritmo", "k", "Sil.", "DB", "CH", "ARI bootstrap", "ARI 30 sementes"]
    if n["comparacao"]["kmeans"]["ari_posicoes"] is not None:
        header += ["ARI posições", "NMI posições"]
    out.append(table(header, rows))
    out.append("\nSilhouette por cluster: " + "; ".join(
        f"{ALGO_NAMES[algo]} " + ", ".join(f"{k}: {num(v)}" for k, v in c["silhouette_por_cluster"].items())
        for algo, c in n["comparacao"].items()))

    f = n["friedman"]
    out += ["", "## Teste de Friedman (bootstrap pareado, 100 iterações)", ""]
    out.append("ARI pareado: " + ", ".join(
        f"{ALGO_NAMES[algo]} {num(v)}" for algo, v in f["ari_pareado"].items()))
    out.append(f"\nFriedman: chi2 = {num(f['chi2'], 2)}, p = {sci(f['p'])}, "
               f"H0 {'rejeitada' if f['rejeita_h0'] else 'não rejeitada'}.\n")
    out.append(table(
        ["Par", "ARIs médios", "W", "p (Bonferroni)", "Significativo"],
        [[f"{ALGO_NAMES[w['A']]} x {ALGO_NAMES[w['B']]}", f"{num(w['mean_A'])} x {num(w['mean_B'])}",
          num(w["statistic"], 0), sci(w["p_bonferroni"]), "Sim" if w["significant"] else "Não"]
         for w in f["wilcoxon"]],
    ))

    out += ["", "## Sensibilidade às features (11 x 9, sem OREB e DREB)", ""]
    out.append(table(
        ["Algoritmo", "k (11)", "Sil. (11)", "k (9)", "Sil. (9)", "ARI (11 x 9)"],
        [[ALGO_NAMES[algo], s["k_11"], num(s["silhouette_11"]), s["k_9"], num(s["silhouette_9"]),
          num(s["ari_11_vs_9"])] for algo, s in n["sensibilidade_features"].items()],
    ))

    out += ["", "## Perfis dos clusters (médias na escala original)", ""]
    for algo, profiles in n["perfis"].items():
        feats = [k for k in profiles[0] if k not in ("Cluster", "N_Players")]
        out.append(f"**{ALGO_NAMES[algo]}**\n")
        out.append(table(
            ["Variável"] + [f"Cluster {p['Cluster']} ({p['N_Players']})" for p in profiles],
            [[ft] + [num(p[ft], 3 if ft.endswith("PCT") else 2) for p in profiles] for ft in feats],
        ))
        out.append("\nMais típicos (mais próximos do centroide):")
        for cluster, names in n["exemplos"][algo].items():
            out.append(f"- Cluster {cluster}: {', '.join(names)}")
        out.append("")

    if n["posicoes_por_cluster"]:
        out += ["## Posições por cluster", ""]
        for algo, ct in n["posicoes_por_cluster"].items():
            cols = list(next(iter(ct.values())))
            out.append(f"**{ALGO_NAMES[algo]}**\n")
            out.append(table(["Cluster"] + cols, [[k] + [ct[k][c] for c in cols] for k in ct]))
            out.append("")

    out += ["## Jogadores híbridos (GMM, secundária >= 25%)", ""]
    if n["hibridos"]:
        out.append(table(
            ["Jogador", "Comp. principal", "Prob.", "Comp. secundário", "Prob."],
            [[h["Player"], h["Primary_Cluster"], num(h["Primary_Prob"]),
              h["Secondary_Cluster"], num(h["Secondary_Prob"])] for h in n["hibridos"]],
        ))
    else:
        out.append("Nenhum.")

    out += ["", "## Figuras", ""]
    out += [f"### {title}\n\n![{title}](figuras/{name})\n" for name, title in FIGURES]

    path.write_text("\n".join(out) + "\n", encoding="utf-8")
