import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from gensim.models import Word2Vec

from model import GNNEncoder, collate_ego_batch
from features_and_head import PredictionHead

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SEED = 42
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3


def get_splits(labels):
    """Identical logic to train.py -- same seed + stratification -> same split."""
    all_idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(all_idx, test_size=0.3, random_state=SEED, stratify=labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=SEED, stratify=labels[temp_idx])
    return train_idx, val_idx, test_idx


def compute_metrics(true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(true, probs) if len(np.unique(true)) > 1 else float("nan")
    return {
        "auc": auc,
        "f1": f1_score(true, preds, zero_division=0),
        "precision": precision_score(true, preds, zero_division=0),
        "recall": recall_score(true, preds, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Baseline 1: Logistic Regression on handcrafted features only
# ---------------------------------------------------------------------------

def run_logistic_regression(hc_features, labels, train_idx, val_idx, test_idx):
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(hc_features[train_idx], labels[train_idx])
    test_probs = clf.predict_proba(hc_features[test_idx])[:, 1]
    return compute_metrics(labels[test_idx], test_probs)


# ---------------------------------------------------------------------------
# Baseline 2: Node2Vec (random walks + skip-gram) + MLP
# ---------------------------------------------------------------------------

def generate_node2vec_walks(g, num_walks=10, walk_length=20, seed=SEED):
    rng = np.random.default_rng(seed)
    nodes = list(g.nodes())
    walks = []
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            for _ in range(walk_length - 1):
                nbrs = list(g.neighbors(walk[-1]))
                if not nbrs:
                    break
                walk.append(nbrs[rng.integers(len(nbrs))])
            walks.append([str(n) for n in walk])
    return walks


def train_node2vec_embeddings(g, dim=32):
    walks = generate_node2vec_walks(g)
    w2v = Word2Vec(walks, vector_size=dim, window=5, min_count=0, sg=1,
                    workers=1, seed=SEED, epochs=3)
    embeddings = np.zeros((g.number_of_nodes(), dim), dtype=np.float32)
    for node in g.nodes():
        embeddings[node] = w2v.wv[str(node)]
    return embeddings


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def run_node2vec_mlp(node2vec_emb, labels, train_idx, val_idx, test_idx):
    x_train = torch.tensor(node2vec_emb[train_idx], dtype=torch.float32)
    y_train = torch.tensor(labels[train_idx], dtype=torch.float32)
    x_test = torch.tensor(node2vec_emb[test_idx], dtype=torch.float32)

    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum().item(), 1))
    model = SimpleMLP(node2vec_emb.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        probs = model(x_train)
        weight = torch.where(y_train == 1, pos_weight, 1.0)
        loss = F.binary_cross_entropy(probs, y_train, weight=weight)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_probs = model(x_test).numpy()
    return compute_metrics(labels[test_idx], test_probs)


# ---------------------------------------------------------------------------
# Baseline 3: Plain GCN/GAT (graph embedding only, no handcrafted fusion)
# ---------------------------------------------------------------------------

class PlainGNNModel(nn.Module):
    def __init__(self, encoder_type="gat"):
        super().__init__()
        self.encoder = GNNEncoder(in_dim=4, hidden_dim=32, out_dim=32, encoder_type=encoder_type)
        self.head = PredictionHead(in_dim=32)

    def forward(self, x, edge_index, batch, center_idx):
        h_v = self.encoder(x, edge_index, batch, center_idx)
        return self.head(h_v)


def make_batches_plain(indices, ego_networks, batch_size, shuffle):
    indices = list(indices)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        egos = [ego_networks[i] for i in indices[start:start + batch_size]]
        yield collate_ego_batch(egos)


def run_plain_gnn(ego_networks, labels, train_idx, val_idx, test_idx, encoder_type="gat"):
    pos_weight = float((labels[train_idx] == 0).sum() / max((labels[train_idx] == 1).sum(), 1))
    model = PlainGNNModel(encoder_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for _ in range(EPOCHS):
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
    return compute_metrics(np.concatenate(all_true), np.concatenate(all_probs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with open(os.path.join(DATA_DIR, "graph.gpickle"), "rb") as f:
        g = nx.Graph(pickle.load(f))
    with open(os.path.join(DATA_DIR, "ego_networks.pkl"), "rb") as f:
        ego_networks = pickle.load(f)
    hc_data = np.load(os.path.join(DATA_DIR, "handcrafted_features.npz"))
    hc_node_ids, hc_features = hc_data["node_ids"], hc_data["features"]
    id_to_hc_idx = {int(nid): i for i, nid in enumerate(hc_node_ids)}

    labels = np.array([ego["y"] for ego in ego_networks])
    train_idx, val_idx, test_idx = get_splits(labels)
    print(f"Using splits -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    # reindex handcrafted features to match ego_networks order (should already match, but be safe)
    hc_reindexed = np.stack([hc_features[id_to_hc_idx[ego["center"]]] for ego in ego_networks])

    results = {}

    print("\n[1/3] Logistic Regression (handcrafted features only)...")
    results["Logistic Regression"] = run_logistic_regression(
        hc_reindexed, labels, train_idx, val_idx, test_idx
    )

    print("[2/3] Node2Vec + MLP (unsupervised graph embedding only)...")
    node2vec_emb = train_node2vec_embeddings(g, dim=32)
    results["Node2Vec + MLP"] = run_node2vec_mlp(node2vec_emb, labels, train_idx, val_idx, test_idx)

    print("[3/3] Plain GAT (graph embedding only, no handcrafted fusion)...")
    results["Plain GAT"] = run_plain_gnn(ego_networks, labels, train_idx, val_idx, test_idx, "gat")

    print("\n=== Baseline comparison (test set) ===")
    print(f"{'Model':<22} {'AUC':>7} {'F1':>7} {'Precision':>10} {'Recall':>8}")
    for name, m in results.items():
        print(f"{name:<22} {m['auc']:>7.4f} {m['f1']:>7.4f} {m['precision']:>10.4f} {m['recall']:>8.4f}")

    for name, m in results.items():
        assert m["auc"] > 0.4, f"{name} AUC suspiciously low ({m['auc']:.4f}) -- check for a bug"

    print("\nOK: all baselines ran and produced sane metrics.")
    print("Compare these numbers to train.py's Full model test AUC to see the effect of feature fusion.")


if __name__ == "__main__":
    main()
