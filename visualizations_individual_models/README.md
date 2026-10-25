# Individual Model Visualizations
This directory contains clean, standalone, high-resolution figures for each individual model
in the bioacoustic bird species identification benchmark.

## Folder Directory
- **`00_comparative_overview/`**: Multi-model comparison bar charts, ROC-AUC ranking, parameter efficiency scatter, and per-species heatmap.
- **`01_custom_cnn/`**: Baseline 4-block CNN visuals.
- **`02_resnet18/`**: Pretrained ResNet-18 visuals.
- **`03_mobilenet_v3/`**: MobileNetV3-Small with Squeeze-and-Excitation attention visuals.
- **`04_crnn/`**: Bioacoustic CNN + Bidirectional GRU + Temporal Attention visuals.
- **`05_efficientnet_b0/`**: EfficientNet-B0 (MBConv + Compound Scaling) visuals.
- **`06_convnext_tiny/`**: Modernized 7x7 Depthwise Separable ConvNeXt-Tiny visuals.
- **`07_birdnet/`**: Cornell BirdNET Foundation Model benchmark visuals.

## Standard Figures in Each Model's Folder:
1. `01_confusion_matrix.png` — High-resolution confusion matrix heatmap.
2. `02_roc_curve.png` — One-vs-Rest Macro ROC curve with exact AUC and random chance baseline.
3. `03_species_f1_ranking.png` — Horizontal bar chart ranking all 15 species from highest to lowest F1-score.
4. `04_precision_vs_recall.png` — Grouped bar chart comparing Precision vs. Recall across all 15 species.
5. `05_training_curves.png` — Clean 2-panel Loss & Accuracy training convergence curves (for trained models).
6. `06_model_summary_card.png` — Executive KPI scorecard summarizing Top-1, Top-3, F1, ROC-AUC, Precision, and Recall.
