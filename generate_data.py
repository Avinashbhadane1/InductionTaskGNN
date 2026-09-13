import os
import pickle
import random
import numpy as np
import networkx as nx

RANDOM_SEED = 42
N_NODES = 2000          
BA_ATTACH = 5           
N_SEEDS = 15           
CASCADE_PROB = 0.06     
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def build_graph(n_nodes: int, m: int, seed: int) -> nx.Graph:
    """Barabasi-Albert graph: preferential attachment -> hub nodes like real social nets."""
    g = nx.barabasi_albert_graph(n=n_nodes, m=m, seed=seed)
    return g


def simulate_independent_cascade(g: nx.Graph, n_seeds: int, prob: float, seed: int):
    """
    Runs an Independent Cascade simulation.

    Returns:
        action_state: dict {node_id: 0 or 1}
        activation_time: dict {node_id: step at which it activated, -1 if never}
    """
    rng = random.Random(seed)
    nodes = list(g.nodes())
    seeds = rng.sample(nodes, n_seeds)

    action_state = {n: 0 for n in nodes}
    activation_time = {n: -1 for n in nodes}

    for s in seeds:
        action_state[s] = 1
        activation_time[s] = 0

    newly_active = set(seeds)
    t = 0
    while newly_active:
        t += 1
        next_active = set()
        for u in newly_active:
            for v in g.neighbors(u):
                if action_state[v] == 0 and rng.random() < prob:
                    action_state[v] = 1
                    activation_time[v] = t
                    next_active.add(v)
        newly_active = next_active

    return action_state, activation_time


def build_user_attributes(g: nx.Graph, seed: int) -> np.ndarray:

    rng = np.random.default_rng(seed)
    n = g.number_of_nodes()
    activity_level = rng.beta(2, 5, size=n)       
    historical_freq = rng.beta(2, 5, size=n)
    account_age_norm = rng.uniform(0, 1, size=n)
    return np.stack([activity_level, historical_freq, account_age_norm], axis=1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    g = build_graph(N_NODES, BA_ATTACH, RANDOM_SEED)
    action_state, activation_time = simulate_independent_cascade(
        g, N_SEEDS, CASCADE_PROB, RANDOM_SEED
    )
    user_attrs = build_user_attributes(g, RANDOM_SEED)

    n_active = sum(action_state.values())
    print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print(f"Cascade result: {n_active} / {N_NODES} nodes activated "
          f"({100 * n_active / N_NODES:.1f}%)")

    if n_active < 50 or n_active > N_NODES * 0.9:
        print("WARNING: activation rate looks extreme (too rare or nearly everyone). "
              "Consider tuning CASCADE_PROB or N_SEEDS.")

    with open(os.path.join(OUT_DIR, "graph.gpickle"), "wb") as f:
        pickle.dump(g, f)

    node_ids = np.array(list(g.nodes()))
    y = np.array([action_state[n] for n in node_ids])
    act_time = np.array([activation_time[n] for n in node_ids])

    np.savez(
        os.path.join(OUT_DIR, "node_data.npz"),
        node_ids=node_ids,
        y=y,
        activation_time=act_time,
        user_attrs=user_attrs,
    )

    print(f"Saved graph to {OUT_DIR}/graph.gpickle")
    print(f"Saved labels/attrs to {OUT_DIR}/node_data.npz")


if __name__ == "__main__":
    main()