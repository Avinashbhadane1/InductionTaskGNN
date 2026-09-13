import os
import pickle
import numpy as np
import networkx as nx
import torch
import torch.nn as nn

from model import GNNEncoder, collate_ego_batch

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# Handcrafted feature extraction (uses the FULL graph, not the truncated ego net)
# ---------------------------------------------------------------------------

def compute_handcrafted_features(g: nx.Graph, node_ids: np.ndarray, y: np.ndarray,
                                  user_attrs: np.ndarray) -> np.ndarray:

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    y_by_id = {nid: y[id_to_idx[nid]] for nid in node_ids}

    degrees = dict(g.degree())
    clustering = nx.clustering(g)
    max_degree = max(degrees.values()) if degrees else 1

    n = len(node_ids)
    feats = np.zeros((n, 7), dtype=np.float32)

    for i, v in enumerate(node_ids):
        neighbors = list(g.neighbors(int(v)))
        total_nbrs = len(neighbors)
        active_nbrs = sum(y_by_id[nb] for nb in neighbors) if total_nbrs > 0 else 0
        influence_ratio = (active_nbrs / total_nbrs) if total_nbrs > 0 else 0.0

        feats[i, 0:3] = user_attrs[i]
        feats[i, 3] = degrees[int(v)] / max_degree if max_degree > 0 else 0.0
        feats[i, 4] = clustering[int(v)]
        feats[i, 5] = active_nbrs / 10.0  # rough normalization, dataset-dependent
        feats[i, 6] = influence_ratio

    return feats


# ---------------------------------------------------------------------------
# Prediction head + full model
# ---------------------------------------------------------------------------

class PredictionHead(nn.Module):
    """Small MLP -> sigmoid, per PS 2.5."""
    def __init__(self, in_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


class FullModel(nn.Module):

    def __init__(self, gnn_in_dim=4, gnn_hidden=32, gnn_out=32,
                 handcrafted_dim=7, encoder_type="gat"):
        super().__init__()
        self.encoder = GNNEncoder(gnn_in_dim, gnn_hidden, gnn_out, encoder_type)
        self.head = PredictionHead(gnn_out + handcrafted_dim)

    def forward(self, x, edge_index, batch, center_idx, handcrafted_feats):
        h_v = self.encoder(x, edge_index, batch, center_idx)
        fused = torch.cat([h_v, handcrafted_feats], dim=-1)
        return self.head(fused)


if __name__ == "__main__":
    with open(os.path.join(DATA_DIR, "graph.gpickle"), "rb") as f:
        g = nx.Graph(pickle.load(f))
    node_data = np.load(os.path.join(DATA_DIR, "node_data.npz"))
    node_ids, y, user_attrs = node_data["node_ids"], node_data["y"], node_data["user_attrs"]

    handcrafted = compute_handcrafted_features(g, node_ids, y, user_attrs)
    print(f"Handcrafted features shape: {handcrafted.shape}")
    print(f"Sample row (node {node_ids[0]}): {handcrafted[0]}")
    assert not np.isnan(handcrafted).any(), "NaNs in handcrafted features!"

    np.savez(os.path.join(DATA_DIR, "handcrafted_features.npz"),
             node_ids=node_ids, features=handcrafted)
    print(f"Saved handcrafted features to {DATA_DIR}/handcrafted_features.npz")

    with open(os.path.join(DATA_DIR, "ego_networks.pkl"), "rb") as f:
        ego_networks = pickle.load(f)

    batch_egos = ego_networks[:64]
    x, edge_index, batch, center_idx, y_true = collate_ego_batch(batch_egos)

    id_to_idx = {int(nid): i for i, nid in enumerate(node_ids)}
    hc_batch = torch.tensor(
        np.stack([handcrafted[id_to_idx[ego["center"]]] for ego in batch_egos]),
        dtype=torch.float32,
    )

    model = FullModel(encoder_type="gat")
    y_hat = model(x, edge_index, batch, center_idx, hc_batch)

    print(f"y_hat shape: {tuple(y_hat.shape)}")
    print(f"y_hat range: [{y_hat.min().item():.4f}, {y_hat.max().item():.4f}]")
    assert y_hat.shape == (64,)
    assert (y_hat >= 0).all() and (y_hat <= 1).all()
    assert not torch.isnan(y_hat).any()
    print("OK: FullModel produces valid probabilities in [0, 1] with no NaNs.")
