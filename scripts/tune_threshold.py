import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from model import collate_ego_batch
from features_and_head import FullModel
from baselines import get_splits

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SEED = 42
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3


def make_batches(indices, ego_networks, hc_features, id_to_hc_idx, batch_size, shuffle):
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


def get_probs(model, indices, ego_networks, hc_features, id_to_hc_idx):
    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for x, edge_index, batch, center_idx, hc, y in make_batches(
            indices, ego_networks, hc_features, id_to_hc_idx, BATCH_SIZE, shuffle=False
        ):
            all_probs.append(model(x, edge_index, batch, center_idx, hc).numpy())
            all_true.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_true)


def metrics_at_threshold(true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    return {
        "f1": f1_score(true, preds, zero_division=0),
        "precision": precision_score(true, preds, zero_division=0),
        "recall": recall_score(true, preds, zero_division=0),
    }


def find_best_threshold(true, probs):
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.05, 0.95, 19):
        f1 = f1_score(true, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with open(os.path.join(DATA_DIR, "ego_networks.pkl"), "rb") as f:
        ego_networks = pickle.load(f)
    hc_data = np.load(os.path.join(DATA_DIR, "handcrafted_features.npz"))
    hc_node_ids, hc_features = hc_data["node_ids"], hc_data["features"]
    id_to_hc_idx = {int(nid): i for i, nid in enumerate(hc_node_ids)}

    labels = np.array([ego["y"] for ego in ego_networks])
    train_idx, val_idx, test_idx = get_splits(labels)

    num_pos = labels[train_idx].sum()
    num_neg = len(train_idx) - num_pos
    raw_ratio = num_neg / max(num_pos, 1)
    soft_weight = np.sqrt(raw_ratio)
    print(f"Raw class weight would be {raw_ratio:.2f}x -- using softened weight {soft_weight:.2f}x instead")

    model = FullModel(encoder_type="gat")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for x, edge_index, batch, center_idx, hc, y in make_batches(
            train_idx, ego_networks, hc_features, id_to_hc_idx, BATCH_SIZE, shuffle=True
        ):
            optimizer.zero_grad()
            probs = model(x, edge_index, batch, center_idx, hc)
            weight = torch.where(y == 1, soft_weight, 1.0)
            loss = F.binary_cross_entropy(probs, y, weight=weight)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d} | train_loss: {np.mean(losses):.4f}")

    # --- threshold tuning on VALIDATION set only ---
    val_probs, val_true = get_probs(model, val_idx, ego_networks, hc_features, id_to_hc_idx)
    best_threshold, best_val_f1 = find_best_threshold(val_true, val_probs)
    print(f"\nBest threshold found on validation set: {best_threshold:.2f} (val F1 = {best_val_f1:.4f})")

    # --- final evaluation on TEST set, comparing default 0.5 vs tuned threshold ---
    test_probs, test_true = get_probs(model, test_idx, ego_networks, hc_features, id_to_hc_idx)
    test_auc = roc_auc_score(test_true, test_probs)
    default_metrics = metrics_at_threshold(test_true, test_probs, 0.5)
    tuned_metrics = metrics_at_threshold(test_true, test_probs, best_threshold)

    print(f"\nTest AUC (threshold-independent): {test_auc:.4f}")
    print(f"\n{'Threshold':<12} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print(f"{'0.50 (old)':<12} {default_metrics['f1']:>8.4f} "
          f"{default_metrics['precision']:>10.4f} {default_metrics['recall']:>8.4f}")
    print(f"{best_threshold:.2f} (tuned)  {tuned_metrics['f1']:>8.4f} "
          f"{tuned_metrics['precision']:>10.4f} {tuned_metrics['recall']:>8.4f}")

    assert tuned_metrics["f1"] >= default_metrics["f1"], "Tuned threshold should not be worse than 0.5"
    print("\nOK: tuned threshold matches or beats the default 0.5 cutoff on F1.")


if __name__ == "__main__":
    main()
