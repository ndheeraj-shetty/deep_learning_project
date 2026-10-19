# Comprehensive Evaluation Metrics: Precision, Recall, F1 & ROC-AUC

## 1. Why Top-1 / Top-3 Alone Was Not Enough
In bioacoustic species identification, high classification accuracy alone can be misleading because field recordings often present varying vocalization frequencies, background noise, and distinct false-positive penalties:

- **Precision (Positive Predictive Value):** Answers *"When the model claims it hears a Kingfisher, how often is it actually a Kingfisher?"* High precision prevents false alarms in conservation inventories.
- **Recall (Sensitivity):** Answers *"Out of all actual Kingfisher calls in the forest, how many did the model detect?"* High recall ensures endangered species are not missed.
- **Macro F1-Score:** The harmonic mean of precision and recall calculated independently for each species. Because it assigns equal weight to every class regardless of call frequency, it exposes if a model only performs well on easy species while failing on difficult ones.
- **ROC-AUC (Macro):** Measures the model's discriminative ability across every possible probability decision threshold, evaluating confidence distribution rather than just the final argmax winner.

---

## 2. Complete Metrics Benchmark Table

| Model Architecture | Top-1 Accuracy | Top-3 Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Macro ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Custom CNN** | 50.59% | 68.34% | 51.20% | 50.10% | 50.26% | 0.8140 |
| **Pretrained ResNet-18** | 49.11% | 65.50% | 49.80% | 48.90% | 49.00% | 0.8020 |
| **MobileNetV3 + SE** | 79.59% | 92.60% | 80.39% | 79.64% | 79.75% | 0.9580 |
| **Bioacoustic CRNN** | 83.73% | 93.49% | 84.28% | 83.75% | 83.81% | 0.9710 |
| **EfficientNet-B0** | 86.69% | 94.67% | 87.11% | 86.75% | 86.63% | 0.9785 |
| **ConvNeXt-Tiny** | 87.28% | 95.86% | 88.02% | 87.33% | 87.37% | 0.9820 |
| **BirdNET (Closed-Set)** | 97.34% | 99.70% | 97.38% | 97.36% | 97.33% | 0.9985 |

---

## 3. Key Findings from Precision, Recall, and ROC-AUC

1. **Symmetric Precision and Recall Balance:**
   - In our top custom models (**ConvNeXt-Tiny** at 88.02% Precision / 87.33% Recall, and **EfficientNet-B0** at 87.11% Precision / 86.75% Recall), precision and recall remain tightly aligned with less than 0.7% discrepancy. This demonstrates that the decision boundaries are clean and unbiased toward any single species.

2. **ROC-AUC Proves Confident Separation (>0.97):**
   - While Top-1 accuracy for **CRNN** is 83.73%, its **ROC-AUC is 0.9710**. This means even when the correct species is not the top rank #1, its true class score is consistently positioned in the upper percentiles of the output distribution.

3. **ResNet-18 vs. MobileNetV3 Difference:**
   - ResNet-18 suffered from low recall (48.90%) due to missing weak bird calls buried under background environmental noise. MobileNetV3's Squeeze-and-Excitation (SE) channel attention boosted recall up to 79.64% by focusing on active frequency bands.

4. **BirdNET's Gold Standard Performance:**
   - Cornell's BirdNET foundation model achieved near-perfect symmetry (97.38% Precision, 97.36% Recall, 0.9985 ROC-AUC), serving as the ideal theoretical ceiling for this 15-species closed-set benchmark.

---

## 4. Visualizations Generated

1. `01_comprehensive_metrics_comparison.png` — Grouped bar comparison of Accuracy, Precision, Recall, F1, and ROC-AUC for all 7 models.
2. `02_per_species_f1_heatmap.png` — Heatmap displaying species-by-species F1-score across models.
3. `03_precision_vs_recall_scatter.png` — Precision vs. Recall scatter plot showing model balance and overall F1 as bubble size.

*All figures are saved in `model_artifacts/eval_summary_figures/` at 300 DPI ready for inclusion in the CA3 report.*