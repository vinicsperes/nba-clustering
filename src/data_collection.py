"""
Coleta dos dados na NBA API.

Uma consulta ao endpoint leaguedashplayerstats (temporada regular, totais da
temporada, perspectiva Base) fornece tudo o que a análise usa:

- Filtro: minutos totais na temporada. A versão por jogo da API arredonda os
  minutos a uma casa decimal, e MIN x GP erra quem está perto do corte.
- Médias por jogo: total dividido pelo número de jogos, sem o arredondamento
  da versão por jogo. Percentuais: acertos divididos por tentativas; quem não
  tentou nenhum arremesso do tipo fica com 0, como na API.
- Posições: a NBA não publica a posição por temporada e o cadastro atual muda
  com o tempo. data/posicoes.csv congela a posição de cada jogador já
  consultado; os ausentes são buscados no CommonPlayerInfo e acrescentados.

Uso:
    python src/data_collection.py --season 2024-25
    python src/data_collection.py --season 2024-25 --filtro 15mpg
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo, leaguedashplayerstats

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    CLUSTERING_FEATURES,
    DATA_DIR,
    DEFAULT_FILTER,
    MIN_MINUTES,
    MIN_MINUTES_PER_GAME,
    SEASON,
    data_path,
)

POSITIONS_PATH = DATA_DIR / "posicoes.csv"

# Estatísticas de contagem convertidas em média por jogo
COUNT_STATS = ["PTS", "REB", "OREB", "DREB", "AST", "STL", "BLK", "TOV",
               "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA"]
PCT_STATS = {"FG_PCT": ("FGM", "FGA"), "FG3_PCT": ("FG3M", "FG3A"), "FT_PCT": ("FTM", "FTA")}

OUTPUT_COLUMNS = (
    ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE", "GP", "MIN", "MIN_TOTAL"]
    + CLUSTERING_FEATURES
    + ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "POSITION", "POSITION_SIMPLE"]
)


def fetch_season_totals(season: str = SEASON) -> pd.DataFrame:
    """Totais da temporada regular de todos os jogadores que entraram em quadra."""
    print(f"Coletando totais da temporada regular {season}...")
    df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="Totals",
        measure_type_detailed_defense="Base",
        timeout=60,
    ).get_data_frames()[0]
    print(f"  {len(df)} jogadores")
    return df


def to_per_game(totals: pd.DataFrame) -> pd.DataFrame:
    """Converte os totais em médias por jogo e recalcula os percentuais."""
    df = totals[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE", "GP"]].copy()
    df["MIN_TOTAL"] = totals["MIN"]
    df["MIN"] = totals["MIN"] / totals["GP"]
    for stat in COUNT_STATS:
        df[stat] = totals[stat] / totals["GP"]
    for pct, (made, att) in PCT_STATS.items():
        df[pct] = (totals[made] / totals[att]).where(totals[att] > 0, 0.0)
    return df


def apply_filter(df: pd.DataFrame, filtro: str = DEFAULT_FILTER) -> pd.DataFrame:
    """Aplica o recorte de inclusão de jogadores."""
    if filtro == "500min":
        mask = df["MIN_TOTAL"] >= MIN_MINUTES
        rule = f">= {MIN_MINUTES} minutos totais"
    elif filtro == "15mpg":
        mask = df["MIN"] >= MIN_MINUTES_PER_GAME
        rule = f">= {MIN_MINUTES_PER_GAME} minutos por jogo"
    else:
        raise ValueError(f"Filtro desconhecido: {filtro}")
    print(f"  Filtro ({rule}): {len(df)} -> {mask.sum()} jogadores")
    return df[mask].reset_index(drop=True)


def _normalize_position(pos: str) -> str:
    """Converte formato longo de posição (Guard, Forward, Center) para curto (G, F, C)."""
    if not pos or pd.isna(pos):
        return "Unknown"
    mapping = {
        "Guard": "G", "Forward": "F", "Center": "C",
        "Guard-Forward": "G-F", "Forward-Guard": "F-G",
        "Forward-Center": "F-C", "Center-Forward": "C-F",
    }
    return mapping.get(str(pos).strip(), str(pos).strip())


def simplify_position(pos: str) -> str:
    """Mapeia posição para uma das 3 categorias tradicionais: Guard, Forward, Center."""
    if not pos or pd.isna(pos) or pos == "Unknown":
        return "Unknown"
    pos = str(pos).strip().upper()
    if pos.startswith("G"):
        return "Guard"
    if pos.startswith("F"):
        return "Forward"
    if pos.startswith("C"):
        return "Center"
    return "Unknown"


def get_positions(players: pd.DataFrame) -> pd.Series:
    """
    Posição de cada jogador, lida de data/posicoes.csv. Jogadores que ainda
    não estão no arquivo são buscados no CommonPlayerInfo e acrescentados.
    """
    if POSITIONS_PATH.exists():
        known = pd.read_csv(POSITIONS_PATH)
    else:
        known = pd.DataFrame(columns=["PLAYER_ID", "PLAYER_NAME", "POSITION"])

    missing = players[~players["PLAYER_ID"].isin(known["PLAYER_ID"])]
    if len(missing):
        print(f"  Buscando a posição de {len(missing)} jogadores no CommonPlayerInfo...")
        new = []
        for _, row in missing.iterrows():
            try:
                info = commonplayerinfo.CommonPlayerInfo(
                    player_id=row["PLAYER_ID"], timeout=60
                ).get_data_frames()[0]
                pos = _normalize_position(info["POSITION"].iloc[0]) if len(info) else "Unknown"
            except Exception as e:
                # Falha da API não é gravada: o jogador é buscado de novo na próxima coleta
                warnings.warn(f"CommonPlayerInfo falhou para {row['PLAYER_NAME']}: {e}")
                pos = None
            if pos is not None:
                new.append({"PLAYER_ID": row["PLAYER_ID"],
                            "PLAYER_NAME": row["PLAYER_NAME"], "POSITION": pos})
            time.sleep(0.6)
        known = pd.concat([known, pd.DataFrame(new)], ignore_index=True)
        known.sort_values("PLAYER_NAME").to_csv(POSITIONS_PATH, index=False)
        print(f"  {POSITIONS_PATH.name} atualizado ({len(known)} jogadores)")

    return players["PLAYER_ID"].map(known.set_index("PLAYER_ID")["POSITION"]).fillna("Unknown")


def collect_season(
    season: str = SEASON,
    filtro: str = DEFAULT_FILTER,
    save: bool = True,
) -> pd.DataFrame:
    """Coleta, filtra e salva o CSV de uma temporada."""
    df = to_per_game(fetch_season_totals(season))
    df = apply_filter(df, filtro)
    df["POSITION"] = get_positions(df)
    df["POSITION_SIMPLE"] = df["POSITION"].apply(simplify_position)
    df = df[OUTPUT_COLUMNS]

    if save:
        path = data_path(season, filtro)
        df.to_csv(path, index=False)
        print(f"  Dados salvos em {path}")
    return df


def load_data(season: str = SEASON, filtro: str = DEFAULT_FILTER) -> pd.DataFrame:
    """Carrega o CSV de dados previamente coletados."""
    path = data_path(season, filtro)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            f"Colete com: python src/data_collection.py --season {season} --filtro {filtro}"
        )
    df = pd.read_csv(path)
    print(f"Dados carregados: {len(df)} jogadores ({path.name})")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coleta os dados de uma temporada.")
    parser.add_argument("--season", default=SEASON)
    parser.add_argument("--filtro", default=DEFAULT_FILTER)
    args = parser.parse_args()
    collect_season(args.season, args.filtro)
