# Social Influence Prediction (GNN)

Predicts whether a user will take an action (adopting a product, reposting something, clicking an ad) based on what their friends/neighbors have already done. Sample a small neighborhood around a user, encode it with a graph neural net, mix in some hand-picked social features, and pass it through a classifier.

Loosely based on the DeepInf paper. No `torch_geometric` — GCN/GAT layers are written from scratch with basic scatter operations, mainly to dodge the torch-scatter/torch-sparse version headaches.

## How it works

1. Generate a synthetic social graph and simulate an action spreading through it (independent cascade model) — this gives us ground-truth labels.
2. For every user, pull out their local k-hop neighborhood.
3. Run that neighborhood through a GNN encoder (GCN or GAT) to get an embedding.
4. Concatenate that embedding with some handcrafted features — degree, clustering coefficient, fraction of active neighbors, etc.
5. Feed it all into a small MLP that outputs a probability.

There's also a Logistic Regression baseline, a Node2Vec + MLP baseline, and a plain-GNN-without-handcrafted-features version, so the full model has something to be compared against.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install torch networkx numpy scikit-learn gensim
```

## Running it

In order, first time through:

```bash
python generate_data.py          # builds the graph + cascade labels
python sample_ego_networks.py    # extracts each user's local neighborhood
python model.py                  # quick check the GNN encoder works
python features_and_head.py      # handcrafted features + fusion model
python train.py                  # trains the full model
python baselines.py              # Logistic Regression / Node2Vec / plain GAT
python ablation.py               # feature ablations + k-hop sensitivity
python tune_threshold.py         # better precision/recall tradeoff
```

Then whenever you retrain or tweak something:

```bash
python generate_report.py        # regenerates report.md with fresh numbers
```

## Structure

```
scripts/
├── generate_data.py
├── sample_ego_networks.py
├── model.py
├── features_and_head.py
├── train.py
├── baselines.py
├── ablation.py
├── tune_threshold.py
├── generate_report.py
├── data/            (generated)
├── models/          (generated)
├── report.md        (generated)
└── results_history.json (generated)
```

## Results

Numbers, baseline comparisons, and ablation results are in [`report.md`](./scripts/report.md) — it gets rewritten every time `generate_report.py` runs, so it always matches whatever's currently trained.

## Notes

- Only about 6.7% of users actually take the action in this synthetic dataset, so training uses a softened class weight and the decision threshold gets tuned on the validation set instead of just using 0.5.
- Ego networks are capped at a fixed size, so going beyond 2 hops usually doesn't add anything — the neighborhood's already full.
