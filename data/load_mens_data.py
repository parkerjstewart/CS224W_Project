#!/usr/bin/env python3
"""
compile_top150_and_matches.py

Uses Jeff Sackmann's tennis_atp dataset:
https://github.com/JeffSackmann/tennis_atp

Creates two CSVs:
  1. top150_players.csv  →  top 150 male players by current ATP ranking
  2. top150_matches.csv  →  ≥3000 most recent matches between those players 
     (starting from 2024, then 2023, then 2022)

All player and match columns from the repo are preserved.
"""

import io
import sys
import requests
import pandas as pd


RAW_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"

# Repository files
PLAYERS_URL = f"{RAW_BASE}/atp_players.csv"
RANKINGS_URL = f"{RAW_BASE}/atp_rankings_current.csv"
MATCH_YEARS = [2024, 2023, 2022]
MATCH_URL_TEMPLATE = f"{RAW_BASE}/atp_matches_{{}}.csv"

# Output files
OUT_PLAYERS = "top150_players.csv"
OUT_MATCHES = "matches.csv"
MIN_MATCHES = 5000


def download_csv(url: str) -> pd.DataFrame:
    """Download CSV from GitHub raw URL."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.content.decode("utf-8")), low_memory=False)


def main():
    # === Step 1: Load data ===
    players = download_csv(PLAYERS_URL)
    rankings = download_csv(RANKINGS_URL)

    # Validate columns
    required_rank_cols = {"ranking_date", "rank", "player", "points"}
    required_player_cols = {
        "player_id", "name_first", "name_last", "hand", "dob", "ioc", "height", "wikidata_id"
    }
    if not required_rank_cols.issubset(rankings.columns):
        sys.exit(f"Error: ranking file missing columns {required_rank_cols - set(rankings.columns)}")
    if not required_player_cols.issubset(players.columns):
        sys.exit(f"Error: player file missing columns {required_player_cols - set(players.columns)}")

    # === Step 2: Select top 150 players ===
    rankings = rankings.sort_values("rank", ascending=True).drop_duplicates("player")
    top150 = rankings.head(150).copy()

    # Merge ranking info with player details (preserve all player info)
    merged = pd.merge(top150, players, how="left", left_on="player", right_on="player_id")
    merged.rename(columns={"rank": "current_rank", "points": "current_points"}, inplace=True)

    # === Step 3: Collect matches ===
    top_ids = set(merged["player_id"].astype(int))
    all_matches = []
    total = 0

    for year in MATCH_YEARS:
        url = MATCH_URL_TEMPLATE.format(year)
        try:
            df = download_csv(url)
        except Exception as e:
            continue

        # Keep only matches where both players are in top150
        mask = df["winner_id"].isin(top_ids) & df["loser_id"].isin(top_ids)
        subset = df[mask]
        all_matches.append(subset)
        total += len(subset)
        if total >= MIN_MATCHES:
            break

    if not all_matches:
        sys.exit("No matches found among top players.")
    matches = pd.concat(all_matches, ignore_index=True)

    # === Step 4: Write outputs ===
    merged.to_csv(OUT_PLAYERS, index=False)
    matches.to_csv(OUT_MATCHES, index=False)


if __name__ == "__main__":
    main()
