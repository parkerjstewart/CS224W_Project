import numpy as np
import pandas as pd
import torch

RAW_PLAYERS_CSV = "top150_players.csv"
RAW_MATCHES_CSV = "matches.csv"

SURFACE_MAP = {
    'Hard': 0,
    'Clay': 1,
    'Grass': 2,
}


def parse_player_data() -> pd.DataFrame:
    """Parse player data from a CSV file.
    Saves player features to 'player_features.csv' and country mapping to 'country_mapping.txt'.
    All selected columns are turned into numerical features for graph construction.
    Returns:
        DataFrame with player features.
    """
    df = pd.read_csv(RAW_PLAYERS_CSV)
    
    # Keep only current rank, player ID, hand, dob, ioc, height
    df.drop(
        columns=["ranking_date", "current_points", "player",
                 "wikidata_id", "name_first", "name_last"],
        inplace=True,
    )

    countries = df["ioc"].unique()
    country_to_index = {country_code: index for index, country_code in enumerate(countries)}

    # encode country + hand
    df["country_num"] = df["ioc"].map(country_to_index)
    df["hand"] = df["hand"].map({"R": 0, "L": 1, "U": 2})

    # drop only the original country code; KEEP 'hand'
    df.drop(columns=["ioc"], inplace=True)

    df.set_index("player_id", inplace=True)
    df.sort_index(inplace=True)

    with open("country_mapping.txt", "w") as f:
        for country, index in country_to_index.items():
            f.write(f"{index}\t{country}\n")

    with open("player_features.csv", "w") as f:
        df.to_csv(f)

    return df

def get_edges(train_split: float):
    """
    Build edges with features for ALL matches, and tag each edge with a split:
    - earliest train_split fraction of matches -> 'train'
    - remaining matches -> alternated between 'val' and 'test'
    
    Edges are directed from winner to loser.
    Includes surface, tourney_level, round, best_of, tourney_date, and days_ago
    as features, plus a 'split' column.
    """
    player_df = pd.read_csv("player_features.csv", index_col="player_id")

    matches = pd.read_csv(RAW_MATCHES_CSV)
    matches.sort_values("tourney_date", inplace=True)
    matches = matches.reset_index(drop=True)  # make sure i is 0..N-1

    n_matches = len(matches)
    rows_to_select = int(np.ceil(n_matches * train_split))

    # same "present" date as before: first match AFTER the train window
    present_date = pd.to_datetime(
        matches.iloc[rows_to_select]["tourney_date"],
        format="%Y%m%d",
    )
    print(f"Using present date as {present_date.date()} for days_ago calculation.")

    edges_data = []

    for i, row in matches.iterrows():
        date = pd.to_datetime(row["tourney_date"], format="%Y%m%d")
        days_ago = (present_date - date).days

        winner_idx = player_df.index.get_loc(row["winner_id"])
        loser_idx  = player_df.index.get_loc(row["loser_id"])
        surface_enum = SURFACE_MAP.get(row["surface"])

        # decide split: earliest rows_to_select are train,
        # remainder is alternated val/test so they share the same distribution
        if i < rows_to_select:
            split = "train"
        elif i % 2 == 0:
            split = "val"
        else:
            split = "test"

        edges_data.append({
            "winner_idx":   winner_idx,
            "loser_idx":    loser_idx,
            "surface":      surface_enum,
            "tourney_level": row["tourney_level"],
            "round":        row["round"],
            "best_of":      int(row["best_of"]),
            "days_ago":     days_ago,
            "split":        split,
        })

    edges = pd.DataFrame(edges_data)

    with open("edges.csv", "w") as f:
        edges.to_csv(f, index=False)

    return edges

# TODO: come back and fix this once the players data is parsed correctly
def get_train_val_test_matches(train_split: float, node_embeddings: torch.tensor):
    """
    Splits the matches data into training, validation, and test sets.
    Args:
        train_split: Fraction of data to use for training.
        node_embeddings: Numpy array of node embeddings. One row per player.
    Returns:
        train_x: tensor of training input.
        train_y: tensor of training labels.
        val_x: tensor of validation input.
        val_y: tensor of validation labels.
        test_x: tensor of test input.
        test_y: tensor of test labels.
    """

    train_x = torch.tensor([])
    train_y = torch.tensor([])
    val_x = torch.tensor([])
    val_y = torch.tensor([])
    test_x = torch.tensor([])
    test_y = torch.tensor([])

    # Write each match's surface type and
    matches = pd.read_csv(RAW_MATCHES_CSV)
    matches.sort_values("tourney_date", inplace=True)
    rows_to_select = int(np.ceil(len(matches) * train_split))
    player_df = pd.read_csv("player_features.csv", index_col="player_id")

    for i, row in matches.iterrows():
        winner_idx = player_df.index.get_loc(row["winner_id"])
        loser_idx = player_df.index.get_loc(row["loser_id"])
        surface_enum = SURFACE_MAP.get(row['surface'])

        winner_emb = node_embeddings[winner_idx]
        loser_emb = node_embeddings[loser_idx]

        match_feature = np.concatenate((winner_emb, loser_emb, [surface_enum]))

        if i < rows_to_select:
            train_x.append(match_feature)
            train_y.append(1)  # Winner wins
            train_x.append(np.concatenate((loser_emb, winner_emb, [surface_enum])))
            train_y.append(0)  # Loser loses
        elif i % 2 == 0:
            val_x.append(match_feature)
            val_y.append(1)
            val_x.append(np.concatenate((loser_emb, winner_emb, [surface_enum])))
            val_y.append(0)
        else:
            test_x.append(match_feature)
            test_y.append(1)
            test_x.append(np.concatenate((loser_emb, winner_emb, [surface_enum])))
            test_y.append(0)


    return train_x, train_y, val_x, val_y, test_x, test_y

if __name__ == "__main__":
    parse_player_data()
    get_edges(0.7)