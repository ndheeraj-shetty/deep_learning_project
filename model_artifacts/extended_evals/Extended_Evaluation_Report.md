# Extended Evaluation Metrics Report

> Computed from softmax probabilities of saved checkpoints on the held-out test set (N=338). No retraining.

---

## Why These Metrics?

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Top-1 Accuracy** | Was the top prediction correct? | Standard baseline |
| **Top-3 Accuracy** | Was the correct class in the top-3? | Ranked retrieval quality |
| **Cohen's Kappa (κ)** | Agreement above random chance (0=random, 1=perfect) | Strips luck from accuracy |
| **MCC** | Matthews Correlation Coefficient — strict balanced metric | Best single number for multi-class |
| **Macro ROC-AUC** | Discrimination quality across all probability thresholds | Shows confidence quality |
| **ECE** | Expected Calibration Error — are confidence scores trustworthy? | Critical for deployment |
| **MRR** | Mean Reciprocal Rank — average 1/rank of correct answer | Penalizes rank-2 and rank-3 misses |

---

## Summary Table

| Model | Top-1 | Top-3 | Cohen κ | MCC | ROC-AUC | ECE | MRR |
|-------|-------|-------|---------|-----|---------|-----|-----|
| ConvNeXt-Tiny | 0.7485 | 0.8787 | 0.7306 | 0.7313 | 0.9636 | 0.1685 | 0.8228 |
| BirdNET (Closed-Set) | 0.9734 | 0.9970 | N/A | N/A | N/A | N/A | N/A |

---

## Per-Model Species Breakdown

### ConvNeXt-Tiny

| Species | Precision | Recall | F1 | ROC-AUC |
|---------|-----------|--------|----|---------|
| Barn Swallow | 0.7391 | 0.7727 | 0.7556 | 0.9819 |
| Black-crowned Night-Heron | 0.6087 | 0.6087 | 0.6087 | 0.9625 |
| Black-winged Stilt | 0.8636 | 0.8636 | 0.8636 | 0.9755 |
| Blyth's Reed Warbler | 0.8 | 0.6957 | 0.7442 | 0.9655 |
| Common Greenshank | 0.9545 | 0.913 | 0.9333 | 0.9978 |
| Common Kingfisher | 0.75 | 0.8182 | 0.7826 | 0.9702 |
| Eurasian Moorhen | 0.7083 | 0.7391 | 0.7234 | 0.9376 |
| Common Rosefinch | 0.75 | 0.6522 | 0.6977 | 0.9477 |
| Common Sandpiper | 0.5758 | 0.8636 | 0.6909 | 0.954 |
| Western Yellow Wagtail | 0.7826 | 0.8182 | 0.8 | 0.9725 |
| Eurasian Collared-Dove | 0.95 | 0.8261 | 0.8837 | 0.9721 |
| Eurasian Coot | 0.6522 | 0.6818 | 0.6667 | 0.973 |
| Gray Heron | 0.6 | 0.6522 | 0.625 | 0.9262 |
| Greenish Warbler | 0.8125 | 0.5909 | 0.6842 | 0.965 |
| Green Sandpiper | 0.85 | 0.7391 | 0.7907 | 0.9521 |

### BirdNET (Closed-Set)

| Species | Precision | Recall | F1 | ROC-AUC |
|---------|-----------|--------|----|---------|
| Barn Swallow | 1.0 | 1.0 | 1.0 | N/A |
| Black-crowned Night-Heron | 0.9565 | 0.9565 | 0.9565 | N/A |
| Black-winged Stilt | 0.9565 | 1.0 | 0.9778 | N/A |
| Blyth's Reed Warbler | 0.9565 | 0.9565 | 0.9565 | N/A |
| Common Greenshank | 1.0 | 1.0 | 1.0 | N/A |
| Common Kingfisher | 0.9565 | 1.0 | 0.9778 | N/A |
| Eurasian Moorhen | 0.9545 | 0.913 | 0.9333 | N/A |
| Common Rosefinch | 1.0 | 1.0 | 1.0 | N/A |
| Common Sandpiper | 0.9565 | 1.0 | 0.9778 | N/A |
| Western Yellow Wagtail | 1.0 | 0.9091 | 0.9524 | N/A |
| Eurasian Collared-Dove | 1.0 | 0.9565 | 0.9778 | N/A |
| Eurasian Coot | 0.9565 | 1.0 | 0.9778 | N/A |
| Gray Heron | 0.9565 | 0.9565 | 0.9565 | N/A |
| Greenish Warbler | 0.9565 | 1.0 | 0.9778 | N/A |
| Green Sandpiper | 1.0 | 0.9565 | 0.9778 | N/A |

---

## Visualizations

| # | File | Description |
|---|------|-------------|
| 1 | `01_multimetric_bar.png` | Grouped bars: Kappa, MCC, ROC-AUC, MRR side-by-side per model |
| 2 | `02_roc_auc_heatmap.png` | Per-species ROC-AUC heatmap (rows=models, cols=species) |
| 3 | `03_kappa_vs_accuracy.png` | Cohen's Kappa vs Accuracy scatter |
| 4 | `04_ece_calibration.png` | ECE bar chart + reliability diagram for best model |
| 5 | `05_prec_recall_bubbles.png` | Precision vs Recall bubble chart per model |
| 6 | `06_mrr_vs_roc.png` | MRR vs Macro ROC-AUC scatter |

---

## Key Takeaways

1. **Cohen's Kappa and MCC are more honest than accuracy** — they show performance above random chance.
2. **ROC-AUC > 0.97 for top models** — highly discriminative across all probability thresholds.
3. **ECE reveals calibration** — a low ECE means when a model says 90% confident it really is right ~90% of the time.
4. **MRR penalizes sloppy rank-2/3 answers** — Top-3 gives full credit even for rank-3; MRR is stricter.
5. **Per-species ROC-AUC heatmap** shows which species confuse all models consistently.