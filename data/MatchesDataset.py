import os
import shutil
import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset

class MatchesDataset(InMemoryDataset):
    def __init__(self, root, player_emb, transform=None, pre_transform=None):
        """
        Args:
            root: Base directory for PyG to store raw/ and processed/ artifacts.
            player_emb: Tensor [num_players, feat_dim] aligned with player_features.csv order.
            transform: Optional PyG transform applied each time a sample is read (e.g., runtime augmentations).
            pre_transform: Optional PyG transform applied once during processing before caching (e.g., normalize, add encodings).
        """
        self.player_emb = player_emb  # tensor [N, C] from TorchFrame encoder
        super().__init__(root, transform, pre_transform)
        # In torch>=2.6, torch.load defaults to weights_only=True, which blocks
        # loading PyG Data objects. Explicitly set weights_only=False since we
        # trust the locally generated cache.
        self.data, self.slices = torch.load(
            self.processed_paths[0],
            weights_only=False,
        )

    @property
    def raw_file_names(self):
        # expect root/raw/edges.csv to exist otherwise download() is called to generate it.
        return ["edges.csv"]

    @property
    def processed_file_names(self):
        # processed artifacts under <root>/processed/; if missing, process() will be called
        return ["data.pt"]

    def process(self):
        # Read edge data from csv or download it if not present
        edges = pd.read_csv(self.raw_paths[0])

        # node features (detached to avoid saving autograd history)
        x = self.player_emb.detach()

        # winner -> loser indices
        src = torch.from_numpy(edges["winner_idx"].to_numpy()).long()
        dst = torch.from_numpy(edges["loser_idx"].to_numpy()).long()

        # categorical encodings
        surface = torch.from_numpy(edges["surface"].to_numpy()).long()
        surface_oh = F.one_hot(surface, num_classes=3).float()  # [E, 3]

        lvl_values = sorted(edges["tourney_level"].unique())
        lvl2id = {lvl: i for i, lvl in enumerate(lvl_values)}
        lvl_idx = torch.from_numpy(edges["tourney_level"].map(lvl2id).to_numpy()).long()
        lvl_oh = F.one_hot(lvl_idx, num_classes=len(lvl2id)).float()  # [E, L]

        rnd_values = sorted(edges["round"].unique())
        rnd2id = {rnd: i for i, rnd in enumerate(rnd_values)}
        rnd_idx = torch.from_numpy(edges["round"].map(rnd2id).to_numpy()).long()
        rnd_oh = F.one_hot(rnd_idx, num_classes=len(rnd2id)).float()  # [E, R]

        # numeric features (normalized)
        best_of = torch.from_numpy(edges["best_of"].to_numpy()).float().unsqueeze(1)
        best_of = (best_of - best_of.mean()) / (best_of.std() + 1e-6)

        days = torch.from_numpy(edges["days_ago"].to_numpy()).float().unsqueeze(1)
        days = (days - days.mean()) / (days.std() + 1e-6)

        edge_attr = torch.cat([surface_oh, lvl_oh, rnd_oh, best_of, days], dim=1)
        E = edge_attr.size(0)

        # use edges.csv as-is (assumed already symmetric/duplicated upstream)
        edge_index = torch.stack([src, dst], dim=0)
        edge_attr_bidir = edge_attr
        edge_type = torch.zeros(E, dtype=torch.long)

        # split masks (no duplication)
        split_map = {"train": 0, "val": 1, "test": 2}
        edge_split = torch.from_numpy(edges["split"].map(split_map).to_numpy()).long()
        train_mask = edge_split == 0
        val_mask = edge_split == 1
        test_mask = edge_split == 2

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr_bidir,
            edge_type=edge_type,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
    
    def download(self):
        # check to see if we already have <root>/raw/edges.csv
        # TODO: make sure this works properly on all cases when calling from the match_predictions notebook
        raw_edges = self.raw_paths[0]
        raw_dir = self.raw_dir
        os.makedirs(raw_dir, exist_ok=True)

        # If edges.csv already exists in raw/, nothing to do.
        if os.path.exists(raw_edges):
            return

        # Try copying an existing edges.csv from the data directory.
        data_dir = os.path.dirname(os.path.abspath(__file__))
        existing_edges = os.path.join(data_dir, "edges.csv")
        if os.path.exists(existing_edges):
            shutil.copyfile(existing_edges, raw_edges)
            return

        # Otherwise generate edges.csv using the data prep scripts.
        try:
            from data import load_mens_data, parse_data  # type: ignore
        except ImportError as e:
            raise RuntimeError("data prep scripts not found; cannot generate edges.csv") from e

        cwd = os.getcwd()
        os.chdir(data_dir)
        try:
            # Download raw players/matches if missing.
            have_players = os.path.exists("top150_players.csv")
            have_matches = os.path.exists("matches.csv")
            if not (have_players and have_matches):
                load_mens_data.main()

            # Build player_features.csv + country_mapping.txt if missing.
            if not os.path.exists("player_features.csv"):
                parse_data.parse_player_data()

            # Build edges.csv with default split.
            parse_data.get_edges(train_split=0.7)
        finally:
            os.chdir(cwd)

        if not os.path.exists(existing_edges):
            raise RuntimeError("Failed to create edges.csv via data prep scripts.")

        shutil.copyfile(existing_edges, raw_edges)

# Usage:
# player_emb = <from TorchFrame encoder>
# ds = MatchesDataset(root="data/processed_matches", player_emb=player_emb)
# g = ds[0]
