# Deep Learning-Based Automated Bird Species Identification Using Acoustic Signals
**Benchmark:** BirdCLEF 2024 (15 Target Species Multiclass Acoustic Classification)  
**Publication Standard:** Zero Data Leakage (Strict Recording-Level Split), Verified Empirics, Full Reproducibility.

---

## 1. Project Overview & Team Workflow
This repository contains the complete deep learning codebase, data engineering manifests, and model artifacts for automated avian species classification from bioacoustic signals.

### Division of Responsibility
- **Teammate 1 (Current Scope Completed):**
  - Data engineering, STFT, log-Mel spectrogram pipeline & recording-level stratified splitting ([`01_data_preprocessing.ipynb`](code/01_data_preprocessing.ipynb)).
  - Baseline Custom 4-Block CNN training & validation ([`02_custom_cnn.ipynb`](code/02_custom_cnn.ipynb)).
  - Deep Residual Network (ResNet-18) architecture & training ([`03_resnet.ipynb`](code/03_resnet.ipynb)).
  - Activity-aware vocalization detection, multi-window temporal ensembling, and interactive inference engine ([`test_model.py`](code/test_model.py)).
- **Teammate 2 (Next Steps):**
  - Implement Mobile Inverted Bottleneck Convolution Network: **`04_efficientnet.ipynb`** (EfficientNet-B0) using the **exact same manifests** (`train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`) and vocabulary (`species_mapping.json`).
  - Joint Final Evaluation & Model Comparison: **`05_evaluation_and_model_comparison.ipynb`** comparing Custom CNN vs. ResNet-18 vs. EfficientNet on the held-out test set ($N=338$).

---

## 2. Current Benchmark Results (Holdout Test Set $N=338$)
The teammate should compare their EfficientNet results against these verified benchmarks:

| Model Architecture | Inference Windowing Method | Top-1 Test Accuracy | Top-3 Test Accuracy | Macro F1 Score |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline Custom CNN** | Center 5.0s Crop | 50.59% | 68.34% | 50.26% |
| **MobileNetV3 + SE Attention** | Center 5.0s Crop | 66.27% | 84.02% | 66.22% |
| **MobileNetV3 + SE Attention** | Peak-Energy Vocalization Crop | 75.74% | 89.94% | 75.79% |
| **FINAL OPTIMIZED MODEL** | **Top-3 Multi-Window Ensemble** | **79.59%** | **92.60%** | **79.75%** |

---

## 3. Repository Structure & Manifest
```text
deep_learning_project/
├── data_artifacts/                 # Controlled scientific artifacts (DO NOT MODIFY)
│   ├── config/                     # preprocessing_config.json (32 kHz, 5s clip, 128 Mels, N_fft 2048)
│   ├── vocab/                      # species_mapping.json (Taxonomic codes, common names, class indices)
│   └── manifests/                  # Disjoint recording-level partitions (0% leakage)
│       ├── train_manifest.csv      # 1,575 recordings (70%)
│       ├── val_manifest.csv        # 337 recordings (15%)
│       └── test_manifest.csv       # 338 recordings (15%)
├── model_artifacts/                # Trained checkpoints, evaluation reports & training curves
│   ├── custom_cnn/                 # Baseline Custom CNN (best_custom_cnn.pt, curves, reports)
│   └── optimized_mobilenet/        # Final Optimized Attention Model (best_mobilenet_model.pt)
├── sample_dataset/                 # Tabular metadata manifests
│   ├── sample_metadata.csv         # Metadata for 15 species subset
│   └── train_metadata.csv          # Mirror for automated notebook ingestion
└── code/                           # Executable research notebooks & CLI tools
    ├── 01_data_preprocessing.ipynb # Full audio-to-spectrogram preprocessing pipeline
    ├── 02_custom_cnn.ipynb         # Custom CNN baseline training & validation
    ├── 03_resnet.ipynb             # ResNet-18 architecture, transfer learning & training
    ├── 04_model_evaluation.ipynb   # Multi-stage evaluation benchmark & live audio visualizer
    ├── test_model.py               # Unified CLI testing tool
    └── train_mobilenet_attention.py # Training script for attention network
```

---

## 4. How Teammate Sets Up the Audio Dataset

### Option A: Google Drive (Recommended for Google Colab)
The audio dataset (`sample_dataset/audio/`, ~758 MB, 2,250 `.ogg` files) is stored in Google Drive to avoid GitHub repository bloat.
1. Download or mount `sample_dataset` in Google Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
# Point dataset_dir to your Google Drive folder:
# /content/drive/MyDrive/deep_learning_project/sample_dataset
```
2. Notebooks automatically detect Google Colab and search for `sample_dataset/` or Google Drive paths.

### Option B: Local Workstation
1. Place the 15-species audio files inside `sample_dataset/audio/<species_code>/<filename>.ogg`.
2. All notebooks and scripts will automatically discover `sample_dataset/audio/`.

---

## 5. How to Test & Run Predictions
```bash
# 1. Run final holdout test evaluation (338 recordings):
python code/test_model.py --model mobilenet --mode evaluate

# 2. Run baseline Custom CNN evaluation:
python code/test_model.py --model custom_cnn --mode evaluate

# 3. Predict on any raw audio file (with top-3 species confidence):
python code/test_model.py --model mobilenet --mode predict --audio sample_dataset/audio/comkin1/XC110343.ogg
```
