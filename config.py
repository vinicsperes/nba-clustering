"""
Configurações centrais do projeto.

TCC: Análise Comparativa de K-Means, GMM e Clustering Hierárquico
para Identificação de Arquétipos de Jogadores NBA
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

for _dir in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    _dir.mkdir(exist_ok=True)

# =============================================================================
# COLETA DE DADOS
# =============================================================================
SEASON = "2024-25"
MIN_MINUTES = 500

# =============================================================================
# FEATURES DE CLUSTERING
# 11 variáveis de box-score conforme definido no pré-projeto.
# Cobrem pontuação, rebote, assistência, defesa e eficiência.
# =============================================================================
# Aliases mantidos para compatibilidade com o notebook
ALL_FEATURES = CLUSTERING_FEATURES = [
    "PTS",      # Pontos por jogo
    "REB",      # Rebotes totais por jogo
    "OREB",     # Rebotes ofensivos por jogo
    "DREB",     # Rebotes defensivos por jogo
    "AST",      # Assistências por jogo
    "STL",      # Roubos de bola por jogo
    "BLK",      # Tocos por jogo
    "FG_PCT",   # Percentual de arremessos de quadra
    "FG3_PCT",  # Percentual de arremessos de 3 pontos
    "FT_PCT",   # Percentual de lances livres
    "TOV",      # Turnovers por jogo
]
BASIC_FEATURES = CLUSTERING_FEATURES

# =============================================================================
# PARÂMETROS DOS ALGORITMOS
# =============================================================================
KMEANS_CONFIG = {
    "k_range": range(3, 13),
    "n_init": 50,
    "max_iter": 500,
    "random_state": 42,
}

GMM_CONFIG = {
    "k_range": range(3, 13),
    "covariance_type": "full",
    "n_init": 10,
    "max_iter": 200,
    "random_state": 42,
}

HIERARCHICAL_CONFIG = {
    "k_range": range(3, 13),
    "linkage": "ward",
}

# =============================================================================
# VALIDAÇÃO
# =============================================================================
VALIDATION_CONFIG = {
    "bootstrap_iterations": 100,
    "bootstrap_sample_fraction": 0.8,
    "n_seeds": 30,
}

# =============================================================================
# MAPEAMENTO DE POSIÇÕES (análise secundária)
# =============================================================================
POSITION_GROUPS = {
    "Guard": ["PG", "SG", "G", "G-F"],
    "Forward": ["SF", "PF", "F", "F-G", "F-C"],
    "Center": ["C", "C-F"],
}

# =============================================================================
# VISUALIZAÇÃO
# =============================================================================
VIZ_CONFIG = {
    "figsize": (12, 8),
    "dpi": 150,
    "style": "seaborn-v0_8-whitegrid",
    "palette": "husl",
}
