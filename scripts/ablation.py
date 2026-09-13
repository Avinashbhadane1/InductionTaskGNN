import os
import pickle
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from model import GNNEncoder, collate_ego_batch
from features_and_head import PredictionHead, FullModel, compute_handcrafted_features
from baselines import PlainGNNModel, make_batches_plain, get_splits
from sample_ego_networks import build_ego_network

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SEED = 42
BATCH_SIZE = 64
QUICK_EPOCHS = 15  # fewer than train.py's 30 -- fine for relative comparison


def set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def make_batches_full(indices, ego_networks, hc_features, id_to_hc_idx, batch_size, shuffle):
    indices = list(indices)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        egos = [ego_networks[i] for i in indices[start:start + batch_size]]
        x, edge_index, batch, center_idx, y = collate_ego_batch(egos)
        hc = torch.tensor(
            np.stack([hc_features[id_to_hc_idx[ego["center"]]] for ego in egos]),
            dtype=torch.float32,
        )
        yield x, edge_index, batch, center_idx, hc, y


def train_and_eval_full(ego_networks, hc_features, id_to_hc_idx, labels,
                         train_idx, test_idx, use_instance_norm=True, epochs=QUICK_EPOCHS):
    set_seed()
    pos_weight = float((labels[train_idx] == 0).sum() / max((labels[train_idx] == 1).sum(), 1))
    model = FullModel(encoder_type="gat")
    model.encoder.use_instance_norm = use_instance_norm  # toggle for ablation B
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        model.train()
        for x, edge_index, batch, center_idx, hc, y in make_batches_full(
            train_idx, ego_networks, hc_features, id_to_hc_idx, BATCH_SIZE, shuffle=True
        ):
            optimizer.zero_grad()
            probs = model(x, edge_index, batch, center_idx, hc)
            weight = torch.where(y == 1, pos_weight, 1.0)
            loss = F.binary_cross_entropy(probs, y, weight=weight)
            loss.backward()
            optimizer.step()

    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for x, edge_index, batch, center_idx, hc, y in make_batches_full(
            test_idx, ego_networks, hc_features, id_to_hc_idx, BATCH_SIZE, shuffle=False
        ):
            all_probs.append(model(x, edge_index, batch, center_idx, hc).numpy())
            all_true.append(y.numpy())
    probs, true = np.concatenate(all_probs), np.concatenate(all_true)
    return roc_auc_score(true, probs)


def train_and_eval_plain(ego_networks, labels, train_idx, test_idx, epochs=QUICK_EPOCHS):
    set_seed()
    pos_weight = float((labels[train_idx] == 0).sum() / max((labels[train_idx] == 1).sum(), 1))
    model = PlainGNNModel(encoder_type="gat")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        model.train()
        for x, edge_index, batch, center_idx, y in make_batches_plain(
            train_idx, ego_networks, BATCH_SIZE, shuffle=True
        ):
            optimizer.zero_grad()
            probs = model(x, edge_index, batch, center_idx)
            weight = torch.where(y == 1, pos_weight, 1.0)
            loss = F.binary_cross_entropy(probs, y, weight=weight)
            loss.backward()
            optimizer.step()

    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for x, edge_index, batch, center_idx, y in make_batches_plain(
            test_idx, ego_networks, BATCH_SIZE, shuffle=False
        ):
            all_probs.append(model(x, edge_index, batch, center_idx).numpy())
            all_true.append(y.numpy())
    return roc_auc_score(np.concatenate(all_true), np.concatenate(all_probs))


def build_ego_networks_at_k(g, node_ids_all, y_all, attrs_all, id_to_idx, k, max_nodes=30):
    egos = []
    for v in node_ids_all:
        egos.append(build_ego_network(g, int(v), k, max_nodes, y_all, attrs_all, id_to_idx))
    return egos


def main():
    with open(os.path.join(DATA_DIR, "graph.gpickle"), "rb") as f:
        g = nx.Graph(pickle.load(f))
    with open(os.path.join(DATA_DIR, "ego_networks.pkl"), "rb") as f:
        ego_networks_k2 = pickle.load(f)
    node_data = np.load(os.path.join(DATA_DIR, "node_data.npz"))
    node_ids_all, y_arr, attrs_all = node_data["node_ids"], node_data["y"], node_data["user_attrs"]
    hc_data = np.load(os.path.join(DATA_DIR, "handcrafted_features.npz"))
    hc_node_ids, hc_features = hc_data["node_ids"], hc_data["features"]
    id_to_hc_idx = {int(nid): i for i, nid in enumerate(hc_node_ids)}
    id_to_idx = {int(nid): i for i, nid in enumerate(node_ids_all)}
    y_all = {int(nid): y_arr[id_to_idx[int(nid)]] for nid in node_ids_all}

    labels = np.array([ego["y"] for ego in ego_networks_k2])
    train_idx, val_idx, test_idx = get_splits(labels)

    print("=" * 60)
    print("ABLATION A: with vs without handcrafted features")
    print("=" * 60)
    auc_with_hc = train_and_eval_full(ego_networks_k2, hc_features, id_to_hc_idx, labels, train_idx, test_idx)
    auc_without_hc = train_and_eval_plain(ego_networks_k2, labels, train_idx, test_idx)
    print(f"With handcrafted features:    AUC = {auc_with_hc:.4f}")
    print(f"Without handcrafted features: AUC = {auc_without_hc:.4f}")
    print(f"Delta: {auc_with_hc - auc_without_hc:+.4f}\n")

    print("=" * 60)
    print("ABLATION B: with vs without instance normalization")
    print("=" * 60)
    auc_with_norm = train_and_eval_full(ego_networks_k2, hc_features, id_to_hc_idx, labels,
                                         train_idx, test_idx, use_instance_norm=True)
    auc_without_norm = train_and_eval_full(ego_networks_k2, hc_features, id_to_hc_idx, labels,
                                            train_idx, test_idx, use_instance_norm=False)
    print(f"With instance norm:    AUC = {auc_with_norm:.4f}")
    print(f"Without instance norm: AUC = {auc_without_norm:.4f}")
    print(f"Delta: {auc_with_norm - auc_without_norm:+.4f}\n")

    print("=" * 60)
    print("ABLATION C: k-hop sensitivity (ego network radius)")
    print("=" * 60)
    for k in (1, 2, 3):
        egos_k = build_ego_networks_at_k(g, node_ids_all, y_all, attrs_all, id_to_idx, k)
        labels_k = np.array([ego["y"] for ego in egos_k])
        # same split indices are valid since node order is identical across k values
        auc_k = train_and_eval_full(egos_k, hc_features, id_to_hc_idx, labels_k, train_idx, test_idx)
        avg_size = np.mean([len(e["node_ids"]) for e in egos_k])
        print(f"k={k}: avg ego-net size = {avg_size:.1f} nodes, test AUC = {auc_k:.4f}")

    print("\nOK: all three ablation studies completed.")


if __name__ == "__main__":
    main()
