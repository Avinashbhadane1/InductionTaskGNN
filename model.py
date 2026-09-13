import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def collate_ego_batch(ego_list, device="cpu"):

    xs, edge_indices, batch_ids, center_idx, ys = [], [], [], [], []
    node_offset = 0

    for i, ego in enumerate(ego_list):
        n = len(ego["node_ids"])
        action = ego["action_state"].reshape(-1, 1)          # (n, 1)
        attrs = ego["user_attrs"]                              # (n, 3)
        x = np.concatenate([action, attrs], axis=1)            # (n, 4)
        xs.append(x)

        ei = ego["edge_index"] + node_offset
        edge_indices.append(ei)

        batch_ids.append(np.full(n, i, dtype=np.int64))
        center_idx.append(node_offset)  # center is always local index 0
        ys.append(ego["y"])

        node_offset += n

    x = torch.tensor(np.concatenate(xs, axis=0), dtype=torch.float32, device=device)
    edge_index = torch.tensor(np.concatenate(edge_indices, axis=1), dtype=torch.long, device=device)
    batch = torch.tensor(np.concatenate(batch_ids, axis=0), dtype=torch.long, device=device)
    center_idx = torch.tensor(center_idx, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.float32, device=device)

    return x, edge_index, batch, center_idx, y


def add_self_loops(edge_index, num_nodes, device="cpu"):
    """Every node needs an edge to itself, or its own features get lost during aggregation."""
    loop_idx = torch.arange(num_nodes, device=device)
    self_loops = torch.stack([loop_idx, loop_idx], dim=0)
    return torch.cat([edge_index, self_loops], dim=1)


def scatter_add(src, index, dim_size):
    """out[index[i]] += src[i]. Pure-PyTorch replacement for torch_scatter.scatter_add."""
    shape = (dim_size,) + src.shape[1:]
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.index_add_(0, index, src)
    return out


# ---------------------------------------------------------------------------
# Per-instance normalization (PS 2.3)
# ---------------------------------------------------------------------------

class InstanceNorm(nn.Module):
    """
    Normalizes node features to zero mean / unit std WITHIN each ego-network
    sample independently (not across the whole batch, like BatchNorm would).
    This matches "per-instance normalization... before aggregation".
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, batch, num_graphs):
        mean = scatter_add(x, batch, num_graphs) / self._counts(batch, num_graphs)
        mean_per_node = mean[batch]
        centered = x - mean_per_node
        var = scatter_add(centered ** 2, batch, num_graphs) / self._counts(batch, num_graphs)
        std_per_node = torch.sqrt(var[batch] + self.eps)
        return centered / std_per_node

    @staticmethod
    def _counts(batch, num_graphs):
        ones = torch.ones(batch.shape[0], 1, device=batch.device)
        c = scatter_add(ones, batch, num_graphs)
        return c.clamp(min=1.0)


# ---------------------------------------------------------------------------
# GCN layer
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    """
    Standard GCN propagation: h_i' = W * sum_{j in N(i) U {i}} (1/sqrt(d_i * d_j)) h_j
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, num_nodes):
        edge_index = add_self_loops(edge_index, num_nodes, device=x.device)
        row, col = edge_index[0], edge_index[1]  # message flows col -> row

        deg = scatter_add(torch.ones(row.shape[0], 1, device=x.device), row, num_nodes).squeeze(-1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # (num_edges,)

        h = self.lin(x)
        messages = h[col] * norm.unsqueeze(-1)
        out = scatter_add(messages, row, num_nodes)
        return out


# ---------------------------------------------------------------------------
# GAT layer
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """
    Single-head GAT: attention weight e_ij = LeakyReLU(a^T [Wh_i || Wh_j]),
    softmax normalized over each node i's incoming neighbors.
    """
    def __init__(self, in_dim, out_dim, negative_slope=0.2):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.attn = nn.Linear(2 * out_dim, 1)
        self.negative_slope = negative_slope

    def forward(self, x, edge_index, num_nodes):
        edge_index = add_self_loops(edge_index, num_nodes, device=x.device)
        row, col = edge_index[0], edge_index[1]

        h = self.lin(x)
        e = self.attn(torch.cat([h[row], h[col]], dim=-1)).squeeze(-1)
        e = F.leaky_relu(e, self.negative_slope)

        # softmax over neighbors of each target node i (grouped by `row`)
        e_exp = torch.exp(e - e.max())  # simple stability shift (fine at this scale)
        denom = scatter_add(e_exp.unsqueeze(-1), row, num_nodes).squeeze(-1)
        alpha = e_exp / (denom[row] + 1e-16)

        messages = h[col] * alpha.unsqueeze(-1)
        out = scatter_add(messages, row, num_nodes)
        return out


# ---------------------------------------------------------------------------
# Full encoder
# ---------------------------------------------------------------------------

class GNNEncoder(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=32, out_dim=32, encoder_type="gat",
                 use_instance_norm=True):
        super().__init__()
        assert encoder_type in ("gcn", "gat")
        Layer = GCNLayer if encoder_type == "gcn" else GATLayer

        self.use_instance_norm = use_instance_norm
        self.instance_norm = InstanceNorm()
        self.layer1 = Layer(in_dim, hidden_dim)
        self.layer2 = Layer(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch, center_idx):
        num_nodes = x.shape[0]
        num_graphs = int(batch.max().item()) + 1

        if self.use_instance_norm:
            x = self.instance_norm(x, batch, num_graphs)
        x = F.relu(self.layer1(x, edge_index, num_nodes))
        x = self.layer2(x, edge_index, num_nodes)

        h_v = x[center_idx]  # embedding of just the center node per ego network
        return h_v


if __name__ == "__main__":
    with open(os.path.join(DATA_DIR, "ego_networks.pkl"), "rb") as f:
        ego_networks = pickle.load(f)

    batch_egos = ego_networks[:64]
    x, edge_index, batch, center_idx, y = collate_ego_batch(batch_egos)
    print(f"Batched graph: {x.shape[0]} nodes, {edge_index.shape[1]} directed edges, "
          f"{len(batch_egos)} ego networks")

    for enc_type in ("gcn", "gat"):
        encoder = GNNEncoder(in_dim=4, hidden_dim=32, out_dim=32, encoder_type=enc_type)
        h_v = encoder(x, edge_index, batch, center_idx)
        has_nan = torch.isnan(h_v).any().item()
        print(f"[{enc_type.upper()}] output embedding shape: {tuple(h_v.shape)}, "
              f"contains NaN: {has_nan}, sample row[0][:5]: {h_v[0][:5].tolist()}")
        assert h_v.shape == (len(batch_egos), 32)
        assert not has_nan

    print("OK: both encoders produce valid, NaN-free embeddings.")
