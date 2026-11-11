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

# =============================================================================
# DADOS
# Temporada principal e temporada de replicação (temporada regular).
# Filtro de inclusão: pelo menos 500 minutos jogados na temporada, somados
# pelo endpoint de totais. "15mpg" (média de pelo menos 15 minutos por jogo)
# é a variante usada na análise de sensibilidade ao recorte.
# =============================================================================
SEASON = "2024-25"
REPLICATION_SEASON = "2025-26"

MIN_MINUTES = 500
MIN_MINUTES_PER_GAME = 15
FILTERS = ("500min", "15mpg")
DEFAULT_FILTER = "500min"


def _suffix(filtro: str) -> str:
    if filtro not in FILTERS:
        raise ValueError(f"Filtro desconhecido: {filtro}. Opções: {FILTERS}")
    return "" if filtro == DEFAULT_FILTER else f"_{filtro}"


def data_path(season: str, filtro: str = DEFAULT_FILTER) -> Path:
    """CSV de uma temporada, ex.: data/nba_players_2024_25.csv."""
    return DATA_DIR / f"nba_players_{season.replace('-', '_')}{_suffix(filtro)}.csv"


def results_path(season: str, filtro: str = DEFAULT_FILTER) -> Path:
    """Pasta de resultados de uma rodada, ex.: results/2024-25/."""
    return RESULTS_DIR / f"{season}{_suffix(filtro)}"


# =============================================================================
# FEATURES DE CLUSTERING
# 11 variáveis de box-score conforme definido no pré-projeto.
# Cobrem pontuação, rebote, assistência, defesa, eficiência e erros.
# =============================================================================
CLUSTERING_FEATURES = [
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

# Limiar de probabilidade no componente secundário do GMM para considerar
# um jogador híbrido.
HYBRID_THRESHOLD = 0.25

# =============================================================================
# VISUALIZAÇÃO
# =============================================================================
VIZ_CONFIG = {
    "figsize": (12, 8),
    "dpi": 150,
    "style": "seaborn-v0_8-whitegrid",
    "palette": "husl",
}
