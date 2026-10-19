# Deep Learning-Based Automated Bird Species Identification Using Acoustic Signals
**Benchmark:** BirdCLEF 2024 (15 Target Species Multiclass Acoustic Classification)  
**Publication Standard:** Zero Data Leakage (Strict Recording-Level Split), Verified Empirics, Full Reproducibility.

---

## 1. Project Overview & Workflow
This repository contains the complete deep learning codebase, data engineering manifests, and model artifacts for automated avian species classification from bioacoustic signals.

### Implemented Model Architectures:
- **Baseline Custom 4-Block CNN**: Handcrafted CNN with 4 Conv2D + BatchNorm + ReLU + MaxPool blocks (~423k params).
- **Pretrained ResNet-18**: Residual CNN backbone with deep bioacoustic head (~11.2M params).
- **MobileNetV3-Small + SE Attention**: Lightweight inverted bottleneck network with Squeeze-and-Excitation (~1.67M params).
- **Bioacoustic CRNN (CNN + Bi-GRU + Attention)**: 4 Conv2D frequency-pooling blocks + 2-layer Bidirectional GRU with temporal attention pooling (~786k params, trained from scratch).
- **EfficientNet-B0 (MBConv)**: Compound-scaled Mobile Inverted Bottleneck Convolution Network with SiLU activations (~4.3M params).
- **ConvNeXt-Tiny**: Modern pure convolutional network using 7×7 depthwise separable convolutions and LayerNorm (~28.0M params).

---

## 2. Comprehensive Empirical Benchmark (Holdout Test Set $N=338$)
All models were tested on the **exact same untouched recording-level test partition ($N=338$)**:

| Model Architecture | Inference Windowing Method | Parameters | Top-1 Test Accuracy | Top-3 Test Accuracy | Macro F1 Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline Custom CNN** | Center 5.0s Crop | 423,727 | 50.59% | 68.34% | 50.26% |
| **ResNet-18** | Center 5.0s Crop | 11,184,143 | 49.11% | — | 49.00% |
| **MobileNetV3 + SE Attention** | Center 5.0s Crop | 1,674,863 | 66.27% | 84.02% | 66.22% |
| **MobileNetV3 + SE Attention** | Peak-Energy Vocalization Crop | 1,674,863 | 75.74% | 89.94% | 75.79% |
| **MobileNetV3 + SE Attention** | Top-3 Multi-Window Ensemble | 1,674,863 | 79.59% | 92.60% | 79.75% |
| **Bioacoustic CRNN (CNN+GRU)** | **Top-3 Multi-Window Ensemble** | **786,800** | **83.73%** | **93.49%** | **83.81%** |
| **EfficientNet-B0 (MBConv)** | **Top-3 Multi-Window Ensemble** | **4,339,595** | **86.69%** | **94.67%** | **86.63%** |
| **ConvNeXt-Tiny (Modern CNN)** | **Top-3 Multi-Window Ensemble** | **28,021,359** | **87.28%** | **95.86%** | **87.37%** |
| **BirdNET (Cornell / Chemnitz)** | **Open-Set (out of 6,522 species)** | *Foundation Model* | **83.14%** | **94.97%** | — |
| **BirdNET (Cornell / Chemnitz)** | **Closed-Set (Target 15 species)** | *Foundation Model* | **97.34%** | **99.70%** | **97.33%** |

---

## 3. Repository Structure & Manifest
```text
deep_learning_project/
├── data_artifacts/                 # Controlled scientific artifacts
│   ├── config/                     # preprocessing_config.json (32 kHz, 5s clip, 128 Mels, N_fft 2048)
│   ├── vocab/                      # species_mapping.json (Taxonomic codes, common names, class indices)
│   ├── cache/                      # Precomputed Log-Mel Spectrogram tensors (train_cache.pt, etc.)
│   └── manifests/                  # Disjoint recording-level partitions (0% leakage)
│       ├── train_manifest.csv      # 1,575 recordings (70%)
│       ├── val_manifest.csv        # 337 recordings (15%)
│       └── test_manifest.csv       # 338 recordings (15%)
├── model_artifacts/                # Checkpoints, evaluation reports, curves & confusion matrices
│   ├── custom_cnn/                 # Baseline Custom CNN (best_custom_cnn.pt, curves, reports)
│   ├── optimized_resnet18/         # Pretrained ResNet-18 (best_optimized_resnet18.pt)
│   ├── optimized_mobilenet/        # MobileNetV3 + SE Attention (best_mobilenet_model.pt)
│   ├── efficientnet_b0/            # EfficientNet-B0 (best_efficientnet_b0.pt, reports, curves)
│   ├── convnext_tiny/              # ConvNeXt-Tiny (best_convnext_tiny.pt, reports, curves)
│   ├── crnn/                       # Bioacoustic CRNN (best_crnn.pt, reports, curves)
│   ├── birdnet/                    # BirdNET Benchmark (reports, confusion matrix)
│   ├── all_models_summary.csv      # Machine-readable evaluation table
│   └── comparative_model_benchmark.png # Consolidated benchmark bar chart
├── sample_dataset/                 # Audio files and metadata manifests
│   ├── audio/                      # 15 species subdirectories containing 2,250 .ogg audio files
│   ├── sample_metadata.csv         # Metadata for 15 species subset
│   └── train_metadata.csv          # Mirror for automated notebook ingestion
└── code/                           # Executable research notebooks & CLI tools
    ├── 01_data_preprocessing.ipynb # Full audio-to-spectrogram preprocessing pipeline
    ├── 02_custom_cnn.ipynb         # Custom CNN baseline training & validation
    ├── 03_resnet.ipynb             # ResNet-18 architecture, transfer learning & training
    ├── 04_efficientnet.ipynb       # EfficientNet-B0 training, validation & holdout testing
    ├── 05_convnext_and_crnn.ipynb  # ConvNeXt-Tiny & CRNN architectures & evaluations
    ├── 06_evaluation_and_model_comparison.ipynb # Joint comparative benchmark across all 6 models + BirdNET
    ├── evaluate_birdnet.py         # BirdNET benchmark evaluation script
    ├── test_model.py               # Unified CLI testing tool for all architectures
    └── train_and_evaluate_models.py # Complete automated training pipeline
```

---

## 4. How to Test & Run Predictions

```bash
# 1. Run final holdout test evaluation (338 recordings) on any model:
python code/test_model.py --model efficientnet --mode evaluate
python code/test_model.py --model convnext --mode evaluate
python code/test_model.py --model crnn --mode evaluate
python code/test_model.py --model mobilenet --mode evaluate
python code/test_model.py --model custom_cnn --mode evaluate

# 2. Predict on any raw audio file (with top-3 species confidence):
python code/test_model.py --model efficientnet --mode predict --audio sample_dataset/audio/comkin1/XC110343.ogg
python code/test_model.py --model convnext --mode predict --audio sample_dataset/audio/barswa/XC133802.ogg
```
