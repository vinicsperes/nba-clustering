"""
Atualiza notebooks/analysis.ipynb adicionando 3 análises:

  (A) GMM alternativo por Silhouette (logo após a seção do GMM original)
      — pra comparação justa lado a lado
  (B) Análise de sensibilidade às features (11 vs 9, sem OREB/DREB)
  (C) Teste estatístico formal de Friedman + Wilcoxon pareado

Estratégia:
  - Idempotente: identifica células já adicionadas via tag e pula.
  - Append no final do notebook (mantém o conteúdo existente intacto).
  - Não executa as células — você precisa abrir o notebook e rodar tudo
    de novo para regenerar o PDF.

Como rodar:
    venv/bin/python update_notebook.py
"""

from pathlib import Path

import nbformat


NOTEBOOK_PATH = Path(__file__).parent / "notebooks" / "analysis.ipynb"
TAG_SENTINEL = "tcc-update-2026-04-27"


def md(source: str) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_markdown_cell(source.strip() + "\n")
    cell.metadata["tags"] = [TAG_SENTINEL]
    return cell


def code(source: str) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_code_cell(source.strip() + "\n")
    cell.metadata["tags"] = [TAG_SENTINEL]
    return cell


# ----------------------------------------------------------------------------
# (A) GMM alternativo por Silhouette — comparação justa
# ----------------------------------------------------------------------------
SECTION_A_MD = """
## 13. GMM alternativo (seleção por Silhouette)

Re-rodamos o GMM selecionando k pelo Silhouette Score em vez do BIC. Esta
configuração permite uma comparação justa lado a lado com K-Means e
Hierárquico (que selecionam por Silhouette). Reportar as duas variantes
do GMM atende a uma possível objeção de banca quanto à escolha da métrica.

Conforme Géron (2025, p. 275), o BIC é a métrica recomendada para GMM porque
o Silhouette pressupõe clusters esféricos. Mantemos o GMM por BIC como
resultado principal e usamos esta variante por Silhouette como referência
comparativa.
"""

SECTION_A_CODE = """
gmm_sil = GMMClusterer(selection_metric='silhouette')
gmm_sil.fit(X)
gmm_sil_result = gmm_sil.get_best_result()

print(f'\\nGMM (Silhouette) -> k={gmm_sil_result.k}, Silhouette={gmm_sil_result.silhouette:.3f}')
print(f'GMM (BIC)        -> k={gmm_result.k}, Silhouette={gmm_result.silhouette:.3f}')
"""


# ----------------------------------------------------------------------------
# (B) Análise de sensibilidade às features
# ----------------------------------------------------------------------------
SECTION_B_MD = """
## 14. Análise de Sensibilidade às Features

A matriz de correlação mostra que `REB`, `OREB` e `DREB` são quase
redundantes (REB x DREB = 0.96; REB = OREB + DREB por construção). Manter
as três triplica o peso da dimensão "rebote" no cálculo de distância
euclidiana sob StandardScaler.

Re-rodamos o pipeline com 9 features (sem OREB e DREB) e comparamos os
resultados com as 11 features originais. ARI próximo de 1 entre as duas
variantes indica que a conclusão é robusta à remoção das features
redundantes.
"""

SECTION_B_CODE = """
from sklearn.metrics import adjusted_rand_score

FEATURES_9 = [f for f in ALL_FEATURES if f not in {'OREB', 'DREB'}]

X9, _, prep9 = prepare_data(df, features=FEATURES_9)

km9 = KMeansClusterer().fit(X9).get_best_result()
gmm9_bic = GMMClusterer(selection_metric='bic').fit(X9).get_best_result()
gmm9_sil = GMMClusterer(selection_metric='silhouette').fit(X9).get_best_result()
hier9 = HierarchicalClusterer().fit(X9).get_best_result()

sens_rows = [
    {
        'Algoritmo': 'K-Means',
        'k(11)': km_result.k, 'Sil(11)': km_result.silhouette,
        'k(9)': km9.k, 'Sil(9)': km9.silhouette,
        'Delta_Sil': km9.silhouette - km_result.silhouette,
        'ARI(11 vs 9)': adjusted_rand_score(km_result.labels, km9.labels),
    },
    {
        'Algoritmo': 'GMM (BIC)',
        'k(11)': gmm_result.k, 'Sil(11)': gmm_result.silhouette,
        'k(9)': gmm9_bic.k, 'Sil(9)': gmm9_bic.silhouette,
        'Delta_Sil': gmm9_bic.silhouette - gmm_result.silhouette,
        'ARI(11 vs 9)': adjusted_rand_score(gmm_result.labels, gmm9_bic.labels),
    },
    {
        'Algoritmo': 'GMM (Sil)',
        'k(11)': gmm_sil_result.k, 'Sil(11)': gmm_sil_result.silhouette,
        'k(9)': gmm9_sil.k, 'Sil(9)': gmm9_sil.silhouette,
        'Delta_Sil': gmm9_sil.silhouette - gmm_sil_result.silhouette,
        'ARI(11 vs 9)': adjusted_rand_score(gmm_sil_result.labels, gmm9_sil.labels),
    },
    {
        'Algoritmo': 'Hierárquico',
        'k(11)': hier_result.k, 'Sil(11)': hier_result.silhouette,
        'k(9)': hier9.k, 'Sil(9)': hier9.silhouette,
        'Delta_Sil': hier9.silhouette - hier_result.silhouette,
        'ARI(11 vs 9)': adjusted_rand_score(hier_result.labels, hier9.labels),
    },
]

sens_df = pd.DataFrame(sens_rows)
print('Sensibilidade à remoção de OREB e DREB (multicolinearidade):\\n')
print(sens_df.round(4).to_string(index=False))
print('\\nLeitura:')
print('  - Delta_Sil > 0  => Silhouette MELHORA com 9 features')
print('  - ARI ~1.0       => particionamentos quase idênticos (estrutura robusta)')
print('  - ARI < 0.5      => particionamento muda significativamente')
"""


# ----------------------------------------------------------------------------
# (C) Teste estatístico de Friedman + Wilcoxon
# ----------------------------------------------------------------------------
SECTION_C_MD = """
## 15. Teste Estatístico Formal — Friedman + Wilcoxon

Para validar formalmente que K-Means difere dos demais algoritmos, aplicamos
o teste de Friedman não-paramétrico (DEMŠAR, 2006) sobre as ARIs bootstrap
**pareadas** — cada uma das 100 iterações usa a mesma amostra para os três
algoritmos, condição necessária para o teste.

Hipóteses:
- $H_0$: as distribuições de ARI dos algoritmos são equivalentes
- $H_1$: pelo menos um algoritmo difere

Se $p < 0{,}05$, $H_0$ é rejeitada e aplicamos Wilcoxon pareado com correção
de Bonferroni para identificar quais pares diferem.
"""

SECTION_C_CODE = """
from src.validation import (
    compute_paired_bootstrap_aris,
    friedman_test,
    wilcoxon_pairwise,
)

ks_used = {
    'kmeans': km_result.k,
    'gmm': gmm_result.k,
    'hierarchical': hier_result.k,
}

ari_paired = compute_paired_bootstrap_aris(
    X,
    algorithms=['kmeans', 'gmm', 'hierarchical'],
    ks=ks_used,
    n_bootstrap=100,
    seed=42,
)

print('ARIs bootstrap pareadas (resumo das 100 iterações):\\n')
print(ari_paired.describe().round(4).to_string())
"""

SECTION_C_CODE_2 = """
fried = friedman_test(ari_paired)
print('=' * 60)
print('TESTE DE FRIEDMAN')
print('=' * 60)
print(f\"  Estatística (chi^2) : {fried['statistic']:.4f}\")
print(f\"  p-valor             : {fried['p_value']:.6e}\")
print(f\"  n blocos            : {fried['n_blocks']}\")
print(f\"  H0 rejeitada (a=.05): {fried['reject_h0']}\")

print('\\n' + '=' * 60)
print('WILCOXON PAREADO COM BONFERRONI (pós-teste)')
print('=' * 60)
wilc = wilcoxon_pairwise(ari_paired, alpha=0.05)
print(wilc.round(6).to_string(index=False))
"""


def already_updated(nb: nbformat.NotebookNode) -> bool:
    for cell in nb.cells:
        tags = cell.metadata.get("tags", []) or []
        if TAG_SENTINEL in tags:
            return True
    return False


def main() -> None:
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)

    if already_updated(nb):
        print(
            f"O notebook já contém células com a tag '{TAG_SENTINEL}'. "
            "Nada a fazer (script é idempotente)."
        )
        return

    new_cells = [
        md(SECTION_A_MD),
        code(SECTION_A_CODE),
        md(SECTION_B_MD),
        code(SECTION_B_CODE),
        md(SECTION_C_MD),
        code(SECTION_C_CODE),
        code(SECTION_C_CODE_2),
    ]

    nb.cells.extend(new_cells)
    nbformat.write(nb, NOTEBOOK_PATH)
    print(f"OK — adicionadas {len(new_cells)} células ao final de:")
    print(f"  {NOTEBOOK_PATH}")
    print("\nPróximo passo: abrir o notebook, 'Run All', re-exportar o PDF.")


if __name__ == "__main__":
    main()
