# Social Influence Prediction -- Experiment Report

Generated: 2026-09-13 23:26:48

## Dataset
- Nodes: 2000
- Edges: 9975
- Positive rate (fraction of nodes that took the action): 0.067
- Train / Val / Test sizes: 1400 / 300 / 300

## Full Model (GNN encoder + handcrafted features, GAT)
- Class weight used (sqrt of imbalance ratio): 3.73x
- Test AUC-ROC (threshold-independent): **0.7996**
- Best decision threshold (tuned on validation set): 0.25

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.50 (default) | 0.1951 | 0.1905 | 0.2000 |
| 0.25 (tuned) | 0.2419 | 0.1442 | 0.7500 |

## Baseline Comparison (test set, AUC-ROC)
| Model | AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| **Full model (proposed)** | **0.7996** | 0.2419 | 0.1442 | 0.7500 |
| Logistic Regression | 0.7782 | 0.2115 | 0.1310 | 0.5500 |
| Node2Vec + MLP | 0.5912 | 0.1642 | 0.0965 | 0.5500 |
| Plain GAT | 0.7261 | 0.2295 | 0.1373 | 0.7000 |

## Ablation Studies
### With vs without handcrafted features
- With: AUC = 0.8020
- Without: AUC = 0.7639
- Delta: +0.0380

### With vs without instance normalization
- With: AUC = 0.8018
- Without: AUC = 0.7991
- Delta: +0.0027

### k-hop sensitivity
| k | Avg ego-network size (nodes) | Test AUC |
|---|---|---|
| 1 | 10.0 | 0.7518 |
| 2 | 30.0 | 0.8018 |
| 3 | 30.0 | 0.8020 |

## Summary
- Best baseline: Logistic Regression (AUC = 0.7782)
- Full model improves on best baseline by +0.0214 AUC
- Handcrafted feature fusion contributes +0.0380 AUC
- Instance normalization contributes +0.0027 AUC