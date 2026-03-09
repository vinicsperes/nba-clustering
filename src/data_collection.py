"""
Módulo de coleta de dados da NBA API.

Endpoints utilizados:
- leaguedashplayerstats: estatísticas básicas e avançadas por jogo
- leaguedashptstats: tracking data (toques, dribles, tempo de posse)
- leaguedashptdefend: estatísticas defensivas no aro
- playerindex / commonplayerinfo: posições dos jogadores
"""

import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashptstats,
    leaguedashptdefend,
    playerindex,
    commonplayerinfo,
)

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR, SEASON, MIN_MINUTES


def fetch_basic_stats(season: str = SEASON) -> pd.DataFrame:
    """Coleta estatísticas básicas per game (pontos, rebotes, assistências, etc.)."""
    print(f"[1/4] Coletando estatísticas básicas ({season})...")
    time.sleep(1)
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Base",
        )
        df = stats.get_data_frames()[0]
        print(f"      {len(df)} jogadores encontrados")
        return df
    except Exception as e:
        print(f"      ERRO: {e}")
        return pd.DataFrame()


def fetch_advanced_stats(season: str = SEASON) -> pd.DataFrame:
    """Coleta estatísticas avançadas (USG_PCT, TS_PCT, NET_RATING, etc.)."""
    print(f"[2/4] Coletando estatísticas avançadas ({season})...")
    time.sleep(1)
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Advanced",
        )
        df = stats.get_data_frames()[0]
        advanced_cols = [
            "PLAYER_ID", "PLAYER_NAME",
            "OFF_RATING", "DEF_RATING", "NET_RATING",
            "AST_PCT", "AST_TO", "AST_RATIO",
            "OREB_PCT", "DREB_PCT", "REB_PCT",
            "EFG_PCT", "TS_PCT", "USG_PCT",
            "PACE", "PIE",
        ]
        available = [c for c in advanced_cols if c in df.columns]
        print(f"      {len(df)} registros, {len(available)} métricas avançadas")
        return df[available]
    except Exception as e:
        print(f"      ERRO: {e}")
        return pd.DataFrame()


def fetch_tracking_stats(season: str = SEASON) -> pd.DataFrame:
    """Coleta tracking data (toques, dribles, tempo de posse por jogo)."""
    print(f"[3/4] Coletando tracking data ({season})...")
    time.sleep(1)
    try:
        stats = leaguedashptstats.LeagueDashPtStats(
            season=season,
            per_mode_simple="PerGame",
            pt_measure_type="Possessions",
        )
        df = stats.get_data_frames()[0]

        time.sleep(1)
        try:
            stats_speed = leaguedashptstats.LeagueDashPtStats(
                season=season,
                per_mode_simple="PerGame",
                pt_measure_type="SpeedDistance",
            )
            df_speed = stats_speed.get_data_frames()[0]
            speed_cols = ["PLAYER_ID", "DIST_FEET", "DIST_MILES", "AVG_SPEED"]
            available_speed = [c for c in speed_cols if c in df_speed.columns]
            if len(available_speed) > 1:
                df = df.merge(df_speed[available_speed], on="PLAYER_ID", how="left")
        except Exception as e:
            warnings.warn(f"SpeedDistance data não disponível: {e}")

        tracking_cols = [
            "PLAYER_ID", "PLAYER_NAME",
            "TOUCHES", "FRONT_CT_TOUCHES", "TIME_OF_POSS",
            "AVG_SEC_PER_TOUCH", "AVG_DRIB_PER_TOUCH",
            "PTS_PER_TOUCH", "ELBOW_TOUCHES", "POST_TOUCHES",
            "PAINT_TOUCHES", "PTS_PER_ELBOW_TOUCH", "PTS_PER_POST_TOUCH",
            "PTS_PER_PAINT_TOUCH",
        ]
        available = [c for c in tracking_cols if c in df.columns]
        print(f"      {len(df)} registros, {len(available)} métricas de tracking")
        return df[available]
    except Exception as e:
        print(f"      ERRO (tracking pode não estar disponível): {e}")
        return pd.DataFrame()


def fetch_defense_stats(season: str = SEASON) -> pd.DataFrame:
    """Coleta estatísticas defensivas no aro (DEF_RIM_FG_PCT)."""
    print(f"[4/4] Coletando estatísticas defensivas ({season})...")
    time.sleep(1)
    try:
        stats = leaguedashptdefend.LeagueDashPtDefend(
            season=season,
            defense_category="Less Than 6Ft",
            per_mode_simple="PerGame",
        )
        df = stats.get_data_frames()[0]
        defense_cols = ["PLAYER_ID", "PLAYER_NAME", "D_FGA", "D_FGM", "D_FG_PCT"]
        available = [c for c in defense_cols if c in df.columns]
        if available:
            df = df[available].rename(columns={
                "D_FG_PCT": "DEF_RIM_FG_PCT",
                "D_FGA": "DEF_RIM_FGA",
            })
        print(f"      {len(df)} registros defensivos")
        return df
    except Exception as e:
        print(f"      ERRO: {e}")
        return pd.DataFrame()


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


def fetch_player_positions(season: str = SEASON, player_ids: list = None) -> pd.DataFrame:
    """
    Coleta posições dos jogadores via PlayerIndex e CommonPlayerInfo.

    Estratégia:
    1. PlayerIndex — retorna todos os jogadores ativos com posição.
    2. CommonPlayerInfo — busca individual para jogadores não encontrados na etapa 1.
    """
    print(f"[+] Coletando posições dos jogadores ({season})...")
    time.sleep(1)

    positions = {}

    try:
        idx = playerindex.PlayerIndex()
        df = idx.get_data_frames()[0]
        if "PERSON_ID" in df.columns:
            df = df.rename(columns={"PERSON_ID": "PLAYER_ID"})
        if "POSITION" in df.columns:
            for _, row in df.iterrows():
                pos = row["POSITION"]
                if pos and pd.notna(pos) and str(pos).strip():
                    positions[row["PLAYER_ID"]] = str(pos).strip()
            print(f"      PlayerIndex: {len(positions)} jogadores com posição")
    except Exception as e:
        print(f"      AVISO PlayerIndex: {e}")

    if player_ids is not None:
        missing = [pid for pid in player_ids if pid not in positions]
        if missing:
            print(f"      Buscando {len(missing)} jogadores faltantes via CommonPlayerInfo...")
            for i, pid in enumerate(missing):
                try:
                    info = commonplayerinfo.CommonPlayerInfo(player_id=pid)
                    df_info = info.get_data_frames()[0]
                    if "POSITION" in df_info.columns and len(df_info) > 0:
                        positions[pid] = _normalize_position(df_info["POSITION"].iloc[0])
                    time.sleep(0.6)
                except Exception as e:
                    warnings.warn(f"CommonPlayerInfo falhou para player {pid}: {e}")
                if (i + 1) % 10 == 0:
                    print(f"        ... {i + 1}/{len(missing)}")
            print(f"      Total: {len(positions)} jogadores")

    if not positions:
        return pd.DataFrame()

    return pd.DataFrame([
        {"PLAYER_ID": pid, "POSITION": pos} for pid, pos in positions.items()
    ])


def patch_positions(csv_path: Path = None, season: str = SEASON) -> pd.DataFrame:
    """
    Adiciona (ou atualiza) a coluna POSITION no CSV existente
    sem re-coletar todas as estatísticas.
    """
    if csv_path is None:
        csv_path = DATA_DIR / f"nba_players_{season.replace('-', '_')}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)

    if "POSITION" in df.columns:
        df = df.drop(columns=["POSITION"])
        print("Coluna POSITION existente removida, re-buscando...")

    positions = fetch_player_positions(season, player_ids=df["PLAYER_ID"].tolist())

    if positions.empty:
        print("AVISO: não foi possível buscar posições.")
        df["POSITION"] = "Unknown"
        return df

    df = df.merge(positions, on="PLAYER_ID", how="left")
    df["POSITION"] = df["POSITION"].fillna("Unknown")
    df["POSITION_SIMPLE"] = df["POSITION"].apply(simplify_position)

    df.to_csv(csv_path, index=False)
    n_known = (df["POSITION"] != "Unknown").sum()
    print(f"POSITION adicionada: {n_known}/{len(df)} jogadores com posição conhecida")
    print(f"CSV atualizado: {csv_path}")

    return df


def _merge_all_stats(
    df_basic: pd.DataFrame,
    df_advanced: pd.DataFrame,
    df_tracking: pd.DataFrame,
    df_defense: pd.DataFrame,
) -> pd.DataFrame:
    if df_basic.empty:
        raise ValueError("Dados básicos não disponíveis.")

    df = df_basic.copy()

    for label, df_extra in [("Advanced", df_advanced), ("Tracking", df_tracking), ("Defense", df_defense)]:
        if not df_extra.empty:
            extra_cols = [c for c in df_extra.columns if c not in df.columns or c == "PLAYER_ID"]
            if len(extra_cols) > 1:
                df = df.merge(df_extra[extra_cols], on="PLAYER_ID", how="left")
                print(f"        + {label} merged")

    print(f"        = {len(df)} jogadores, {len(df.columns)} colunas")
    return df


def _compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "FG3A" in df.columns and "FGA" in df.columns:
        df["FG3A_RATE"] = (df["FG3A"] / df["FGA"].clip(lower=0.1)).clip(0, 1).fillna(0)

    for stat, col in [("STL", "STL_PCT"), ("BLK", "BLK_PCT")]:
        if col not in df.columns and stat in df.columns and "MIN" in df.columns:
            df[col] = (df[stat] / df["MIN"].clip(lower=1) * 48 * 100).clip(0, 15)

    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].fillna(0)

    return df


def _filter_players(df: pd.DataFrame, min_minutes: float = MIN_MINUTES) -> pd.DataFrame:
    if "MIN" in df.columns and "GP" in df.columns:
        df["TOTAL_MIN"] = df["MIN"] * df["GP"]
        n_before = len(df)
        df = df[df["TOTAL_MIN"] >= min_minutes].copy()
        print(f"  Filtro de minutos ({min_minutes} min totais): {n_before} → {len(df)} jogadores")
    else:
        print("  AVISO: colunas MIN/GP não encontradas, filtro ignorado.")
    return df.reset_index(drop=True)


def collect_all_data(
    season: str = SEASON,
    min_minutes: float = MIN_MINUTES,
    save: bool = True,
) -> pd.DataFrame:
    """
    Coleta e consolida todas as estatísticas dos jogadores NBA da temporada.

    Etapas:
    1. Estatísticas básicas + avançadas (leaguedashplayerstats)
    2. Tracking data (leaguedashptstats)
    3. Defesa no aro (leaguedashptdefend)
    4. Merge, features derivadas, filtro de minutos, posições

    Parameters
    ----------
    season : str
        Temporada NBA no formato "YYYY-YY" (ex: "2024-25").
    min_minutes : float
        Minutos totais mínimos para inclusão do jogador.
    save : bool
        Se True, salva o CSV em data/.
    """
    print("=" * 60)
    print(f"COLETA DE DADOS NBA — Temporada {season}")
    print("=" * 60)

    df_basic = fetch_basic_stats(season)
    df_advanced = fetch_advanced_stats(season)
    df_tracking = fetch_tracking_stats(season)
    df_defense = fetch_defense_stats(season)

    print("\n[MERGE] Combinando datasets...")
    df = _merge_all_stats(df_basic, df_advanced, df_tracking, df_defense)
    df = _compute_derived_features(df)
    df = _filter_players(df, min_minutes)

    positions = fetch_player_positions(season, player_ids=df["PLAYER_ID"].tolist())
    if not positions.empty:
        df = df.merge(positions, on="PLAYER_ID", how="left")
        df["POSITION"] = df["POSITION"].fillna("Unknown")
        df["POSITION_SIMPLE"] = df["POSITION"].apply(simplify_position)
        n_known = (df["POSITION"] != "Unknown").sum()
        print(f"\n[POSITIONS] {n_known}/{len(df)} jogadores com posição")

    if save:
        output_path = DATA_DIR / f"nba_players_{season.replace('-', '_')}.csv"
        df.to_csv(output_path, index=False)
        print(f"\n[SAVE] Dados salvos em {output_path}")

    print(f"\n{'=' * 60}")
    print(f"COLETA FINALIZADA: {len(df)} jogadores, {len(df.columns)} features")
    print("=" * 60)

    return df


def load_data(season: str = SEASON) -> pd.DataFrame:
    """Carrega o CSV de dados previamente coletados."""
    filepath = DATA_DIR / f"nba_players_{season.replace('-', '_')}.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {filepath}\n"
            "Execute collect_all_data() para coletar os dados."
        )
    df = pd.read_csv(filepath)
    print(f"Dados carregados: {len(df)} jogadores ({filepath.name})")
    return df
