import os
import json
import pickle
from datetime import datetime

import numpy as np
import networkx as nx
import torch

from model import collate_ego_batch
from features_and_head import FullModel, compute_handcrafted_features
from baselines import (
    get_splits, run_logistic_regression, train_node2vec_embeddings,
    run_node2vec_mlp, run_plain_gnn, PlainGNNModel,
)
from ablation import train_and_eval_full, train_and_eval_plain, build_ego_networks_at_k
from tune_threshold import get_probs, find_best_threshold, metrics_at_threshold
from sklearn.metrics import roc_auc_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "report.md")
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "results_history.json")
SEED = 42
BATCH_SIZE = 64
EPOCHS = 30


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


def train_full_model_with_threshold(ego_networks, hc_features, id_to_hc_idx, labels,
                                     train_idx, val_idx, test_idx):
    import torch.nn.functional as F
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    num_pos = labels[train_idx].sum()
    num_neg = len(train_idx) - num_pos
    soft_weight = float(np.sqrt(num_neg / max(num_pos, 1)))

    model = FullModel(encoder_type="gat")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(EPOCHS):
        model.train()
        for x, edge_index, batch, center_idx, hc, y in make_batches(
            train_idx, ego_networks, hc_features, id_to_hc_idx, BATCH_SIZE, shuffle=True
        ):
            optimizer.zero_grad()
            probs = model(x, edge_index, batch, center_idx, hc)
            weight = torch.where(y == 1, soft_weight, 1.0)
            loss = F.binary_cross_entropy(probs, y, weight=weight)
            loss.backward()
            optimizer.step()

    val_probs, val_true = get_probs(model, val_idx, ego_networks, hc_features, id_to_hc_idx)
    best_threshold, _ = find_best_threshold(val_true, val_probs)

    test_probs, test_true = get_probs(model, test_idx, ego_networks, hc_features, id_to_hc_idx)
    test_auc = roc_auc_score(test_true, test_probs)
    m_default = metrics_at_threshold(test_true, test_probs, 0.5)
    m_tuned = metrics_at_threshold(test_true, test_probs, best_threshold)

    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "models", "full_model.pt"))

    return {
        "auc": test_auc,
        "soft_weight": soft_weight,
        "best_threshold": best_threshold,
        "metrics_at_0.5": m_default,
        "metrics_at_tuned": m_tuned,
    }


def run_baselines(g, ego_networks, hc_features, id_to_hc_idx, labels, train_idx, val_idx, test_idx):
    hc_reindexed = np.stack([hc_features[id_to_hc_idx[ego["center"]]] for ego in ego_networks])

    results = {}
    results["Logistic Regression"] = run_logistic_regression(hc_reindexed, labels, train_idx, val_idx, test_idx)

    node2vec_emb = train_node2vec_embeddings(g, dim=32)
    results["Node2Vec + MLP"] = run_node2vec_mlp(node2vec_emb, labels, train_idx, val_idx, test_idx)

    results["Plain GAT"] = run_plain_gnn(ego_networks, labels, train_idx, val_idx, test_idx, "gat")
    return results


def run_ablations(g, ego_networks, hc_features, id_to_hc_idx, node_ids_all, y_all, attrs_all,
                   id_to_idx, labels, train_idx, test_idx):
    results = {}

    auc_with_hc = train_and_eval_full(ego_networks, hc_features, id_to_hc_idx, labels, train_idx, test_idx)
    auc_without_hc = train_and_eval_plain(ego_networks, labels, train_idx, test_idx)
    results["handcrafted_features"] = {"with": auc_with_hc, "without": auc_without_hc}

    auc_with_norm = train_and_eval_full(ego_networks, hc_features, id_to_hc_idx, labels,
                                         train_idx, test_idx, use_instance_norm=True)
    auc_without_norm = train_and_eval_full(ego_networks, hc_features, id_to_hc_idx, labels,
                                            train_idx, test_idx, use_instance_norm=False)
    results["instance_norm"] = {"with": auc_with_norm, "without": auc_without_norm}

    khop_results = {}
    for k in (1, 2, 3):
        egos_k = build_ego_networks_at_k(g, node_ids_all, y_all, attrs_all, id_to_idx, k)
        labels_k = np.array([ego["y"] for ego in egos_k])
        auc_k = train_and_eval_full(egos_k, hc_features, id_to_hc_idx, labels_k, train_idx, test_idx)
        avg_size = float(np.mean([len(e["node_ids"]) for e in egos_k]))
        khop_results[k] = {"auc": auc_k, "avg_ego_size": avg_size}
    results["k_hop_sensitivity"] = khop_results

    return results


def render_report(dataset_stats, full_results, baseline_results, ablation_results, timestamp):
    lines = []
    lines.append("# Social Influence Prediction -- Experiment Report")
    lines.append(f"\nGenerated: {timestamp}\n")

    lines.append("## Dataset")
    lines.append(f"- Nodes: {dataset_stats['n_nodes']}")
    lines.append(f"- Edges: {dataset_stats['n_edges']}")
    lines.append(f"- Positive rate (fraction of nodes that took the action): {dataset_stats['pos_rate']:.3f}")
    lines.append(f"- Train / Val / Test sizes: {dataset_stats['n_train']} / {dataset_stats['n_val']} / {dataset_stats['n_test']}\n")

    lines.append("## Full Model (GNN encoder + handcrafted features, GAT)")
    lines.append(f"- Class weight used (sqrt of imbalance ratio): {full_results['soft_weight']:.2f}x")
    lines.append(f"- Test AUC-ROC (threshold-independent): **{full_results['auc']:.4f}**")
    lines.append(f"- Best decision threshold (tuned on validation set): {full_results['best_threshold']:.2f}\n")
    lines.append("| Threshold | F1 | Precision | Recall |")
    lines.append("|---|---|---|---|")
    d = full_results["metrics_at_0.5"]
    t = full_results["metrics_at_tuned"]
    lines.append(f"| 0.50 (default) | {d['f1']:.4f} | {d['precision']:.4f} | {d['recall']:.4f} |")
    lines.append(f"| {full_results['best_threshold']:.2f} (tuned) | {t['f1']:.4f} | {t['precision']:.4f} | {t['recall']:.4f} |\n")

    lines.append("## Baseline Comparison (test set, AUC-ROC)")
    lines.append("| Model | AUC | F1 | Precision | Recall |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| **Full model (proposed)** | **{full_results['auc']:.4f}** | {t['f1']:.4f} | {t['precision']:.4f} | {t['recall']:.4f} |")
    for name, m in baseline_results.items():
        lines.append(f"| {name} | {m['auc']:.4f} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} |")
    lines.append("")

    lines.append("## Ablation Studies")
    hc = ablation_results["handcrafted_features"]
    lines.append("### With vs without handcrafted features")
    lines.append(f"- With: AUC = {hc['with']:.4f}")
    lines.append(f"- Without: AUC = {hc['without']:.4f}")
    lines.append(f"- Delta: {hc['with'] - hc['without']:+.4f}\n")

    norm = ablation_results["instance_norm"]
    lines.append("### With vs without instance normalization")
    lines.append(f"- With: AUC = {norm['with']:.4f}")
    lines.append(f"- Without: AUC = {norm['without']:.4f}")
    lines.append(f"- Delta: {norm['with'] - norm['without']:+.4f}\n")

    lines.append("### k-hop sensitivity")
    lines.append("| k | Avg ego-network size (nodes) | Test AUC |")
    lines.append("|---|---|---|")
    for k, kr in ablation_results["k_hop_sensitivity"].items():
        lines.append(f"| {k} | {kr['avg_ego_size']:.1f} | {kr['auc']:.4f} |")
    lines.append("")

    lines.append("## Summary")
    best_baseline_name = max(baseline_results, key=lambda n: baseline_results[n]["auc"])
    best_baseline_auc = baseline_results[best_baseline_name]["auc"]
    lines.append(f"- Best baseline: {best_baseline_name} (AUC = {best_baseline_auc:.4f})")
    lines.append(f"- Full model improves on best baseline by {full_results['auc'] - best_baseline_auc:+.4f} AUC")
    lines.append(f"- Handcrafted feature fusion contributes {hc['with'] - hc['without']:+.4f} AUC")
    lines.append(f"- Instance normalization contributes {norm['with'] - norm['without']:+.4f} AUC")

    return "\n".join(lines)


def main():
    with open(os.path.join(DATA_DIR, "graph.gpickle"), "rb") as f:
        g = nx.Graph(pickle.load(f))
    with open(os.path.join(DATA_DIR, "ego_networks.pkl"), "rb") as f:
        ego_networks = pickle.load(f)
    node_data = np.load(os.path.join(DATA_DIR, "node_data.npz"))
    node_ids_all, y_arr, attrs_all = node_data["node_ids"], node_data["y"], node_data["user_attrs"]
    hc_data = np.load(os.path.join(DATA_DIR, "handcrafted_features.npz"))
    hc_node_ids, hc_features = hc_data["node_ids"], hc_data["features"]
    id_to_hc_idx = {int(nid): i for i, nid in enumerate(hc_node_ids)}
    id_to_idx = {int(nid): i for i, nid in enumerate(node_ids_all)}
    y_all = {int(nid): y_arr[id_to_idx[int(nid)]] for nid in node_ids_all}

    labels = np.array([ego["y"] for ego in ego_networks])
    train_idx, val_idx, test_idx = get_splits(labels)

    dataset_stats = {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "pos_rate": float(labels.mean()),
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
    }

    print("[1/3] Training full model + tuning threshold...")
    full_results = train_full_model_with_threshold(
        ego_networks, hc_features, id_to_hc_idx, labels, train_idx, val_idx, test_idx
    )
    print(f"      Full model test AUC: {full_results['auc']:.4f}")

    print("[2/3] Running baselines (Logistic Regression, Node2Vec+MLP, Plain GAT)...")
    baseline_results = run_baselines(g, ego_networks, hc_features, id_to_hc_idx, labels,
                                      train_idx, val_idx, test_idx)

    print("[3/3] Running ablations (handcrafted features, instance norm, k-hop)...")
    ablation_results = run_ablations(g, ego_networks, hc_features, id_to_hc_idx,
                                      node_ids_all, y_all, attrs_all, id_to_idx,
                                      labels, train_idx, test_idx)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_text = render_report(dataset_stats, full_results, baseline_results, ablation_results, timestamp)

    with open(REPORT_PATH, "w") as f:
        f.write(report_text)
    print(f"\nWrote {REPORT_PATH}")

    # append this run's key numbers to a history file so you can track changes over time
    history_entry = {
        "timestamp": timestamp,
        "full_model_auc": full_results["auc"],
        "baseline_aucs": {name: m["auc"] for name, m in baseline_results.items()},
        "ablation_deltas": {
            "handcrafted_features": ablation_results["handcrafted_features"]["with"] - ablation_results["handcrafted_features"]["without"],
            "instance_norm": ablation_results["instance_norm"]["with"] - ablation_results["instance_norm"]["without"],
        },
    }
    history = []
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            history = json.load(f)
    history.append(history_entry)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Appended run to {HISTORY_PATH} ({len(history)} run(s) total)")

    print("\nOK: report.md generated successfully.")


if __name__ == "__main__":
    main()
