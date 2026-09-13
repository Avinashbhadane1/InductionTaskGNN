import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from model import collate_ego_batch
from features_and_head import FullModel

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
ENCODER_TYPE = "gat"
SEED = 42


def load_everything():
    with open(os.path.join(DATA_DIR, "ego_networks.pkl"), "rb") as f:
        ego_networks = pickle.load(f)
    hc_data = np.load(os.path.join(DATA_DIR, "handcrafted_features.npz"))
    hc_node_ids = hc_data["node_ids"]
    hc_features = hc_data["features"]
    id_to_hc_idx = {int(nid): i for i, nid in enumerate(hc_node_ids)}
    return ego_networks, hc_features, id_to_hc_idx


def make_batches(indices, ego_networks, hc_features, id_to_hc_idx, batch_size, shuffle):
    indices = list(indices)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        egos = [ego_networks[i] for i in batch_idx]
        x, edge_index, batch, center_idx, y = collate_ego_batch(egos)
        hc = torch.tensor(
            np.stack([hc_features[id_to_hc_idx[ego["center"]]] for ego in egos]),
            dtype=torch.float32,
        )
        yield x, edge_index, batch, center_idx, hc, y


def evaluate(model, indices, ego_networks, hc_features, id_to_hc_idx):
    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for x, edge_index, batch, center_idx, hc, y in make_batches(
            indices, ego_networks, hc_features, id_to_hc_idx, BATCH_SIZE, shuffle=False
        ):
            probs = model(x, edge_index, batch, center_idx, hc)
            all_probs.append(probs.numpy())
            all_true.append(y.numpy())
    probs = np.concatenate(all_probs)
    true = np.concatenate(all_true)
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(true, probs) if len(np.unique(true)) > 1 else float("nan")
    f1 = f1_score(true, preds, zero_division=0)
    precision = precision_score(true, preds, zero_division=0)
    recall = recall_score(true, preds, zero_division=0)
    return {"auc": auc, "f1": f1, "precision": precision, "recall": recall}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)

    ego_networks, hc_features, id_to_hc_idx = load_everything()
    labels = np.array([ego["y"] for ego in ego_networks])
    all_idx = np.arange(len(ego_networks))

    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.3, random_state=SEED, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=SEED, stratify=labels[temp_idx]
    )
    print(f"Split sizes -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    print(f"Positive rate -> train: {labels[train_idx].mean():.3f}, "
          f"val: {labels[val_idx].mean():.3f}, test: {labels[test_idx].mean():.3f}")

    num_pos = labels[train_idx].sum()
    num_neg = len(train_idx) - num_pos
    pos_weight_value = float(num_neg / max(num_pos, 1))
    print(f"Class weighting: positive examples upweighted by {pos_weight_value:.2f}x")

    model = FullModel(encoder_type=ENCODER_TYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_auc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []
        for x, edge_index, batch, center_idx, hc, y in make_batches(
            train_idx, ego_networks, hc_features, id_to_hc_idx, BATCH_SIZE, shuffle=True
        ):
            optimizer.zero_grad()
            probs = model(x, edge_index, batch, center_idx, hc)
            weight = torch.where(y == 1, pos_weight_value, 1.0)
            loss = F.binary_cross_entropy(probs, y, weight=weight)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        val_metrics = evaluate(model, val_idx, ego_networks, hc_features, id_to_hc_idx)
        print(f"Epoch {epoch:2d} | train_loss: {np.mean(epoch_losses):.4f} | "
              f"val_auc: {val_metrics['auc']:.4f} | val_f1: {val_metrics['f1']:.4f}")

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_idx, ego_networks, hc_features, id_to_hc_idx)
    print(f"\nBest val AUC: {best_val_auc:.4f}")
    print(f"Test metrics -> AUC: {test_metrics['auc']:.4f}, F1: {test_metrics['f1']:.4f}, "
          f"Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "full_model.pt"))
    print(f"Saved trained weights to {MODEL_DIR}/full_model.pt")

    assert test_metrics["auc"] > 0.5, "Model should beat random guessing (AUC > 0.5)"
    print("OK: trained model beats random guessing on held-out test set.")


if __name__ == "__main__":
    main()
