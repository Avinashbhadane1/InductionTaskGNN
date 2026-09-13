import os
import pickle
import numpy as np
import networkx as nx
from collections import deque

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
K_HOPS = 2
MAX_NODES = 30  


def bfs_khop(g: nx.Graph, center, k: int, max_nodes: int):

    visited = {center}
    order = [center]
    frontier = deque([(center, 0)])

    while frontier and len(order) < max_nodes:
        node, dist = frontier.popleft()
        if dist == k:
            continue
        for nbr in g.neighbors(node):
            if nbr not in visited:
                visited.add(nbr)
                order.append(nbr)
                frontier.append((nbr, dist + 1))
                if len(order) >= max_nodes:
                    break
    return order


def build_ego_network(g: nx.Graph, center, k: int, max_nodes: int,
                       y_all: dict, attrs_all: np.ndarray, id_to_idx: dict):
    node_ids = bfs_khop(g, center, k, max_nodes)
    local_index = {nid: i for i, nid in enumerate(node_ids)}  # global id -> local pos

    # induced subgraph edges, reindexed to local positions
    sub = g.subgraph(node_ids)
    edges = list(sub.edges())
    if edges:
        edge_index = np.array(
            [[local_index[u], local_index[v]] for u, v in edges], dtype=np.int64
        ).T
        # make undirected explicit (both directions) -- GNN message passing needs this
        edge_index = np.concatenate([edge_index, edge_index[[1, 0]]], axis=1)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    action_state = np.array([y_all[nid] for nid in node_ids], dtype=np.float32)
    true_y = int(action_state[0])   # center's real label, saved separately
    action_state[0] = -1.0          # mask center's own state -- must not leak into input

    user_attrs = np.stack([attrs_all[id_to_idx[nid]] for nid in node_ids], axis=0)

    return {
        "center": center,
        "node_ids": np.array(node_ids, dtype=np.int64),
        "edge_index": edge_index,
        "action_state": action_state,
        "user_attrs": user_attrs,
        "y": true_y,
    }


def main():
    with open(os.path.join(DATA_DIR, "graph.gpickle"), "rb") as f:
        g = nx.Graph(pickle.load(f))

    node_data = np.load(os.path.join(DATA_DIR, "node_data.npz"))
    node_ids_all = node_data["node_ids"]
    y_arr = node_data["y"]
    attrs_all = node_data["user_attrs"]

    id_to_idx = {nid: i for i, nid in enumerate(node_ids_all)}
    y_all = {nid: y_arr[id_to_idx[nid]] for nid in node_ids_all}

    ego_networks = []
    sizes = []
    for v in node_ids_all:
        ego = build_ego_network(g, int(v), K_HOPS, MAX_NODES, y_all, attrs_all, id_to_idx)
        ego_networks.append(ego)
        sizes.append(len(ego["node_ids"]))

    sizes = np.array(sizes)
    print(f"Built {len(ego_networks)} ego networks (k={K_HOPS}, max_nodes={MAX_NODES})")
    print(f"Ego network size: min={sizes.min()}, mean={sizes.mean():.1f}, max={sizes.max()}")
    n_isolated = int((sizes == 1).sum())
    print(f"Nodes with no reachable neighbors within {K_HOPS} hops: {n_isolated}")

    out_path = os.path.join(DATA_DIR, "ego_networks.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(ego_networks, f)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()