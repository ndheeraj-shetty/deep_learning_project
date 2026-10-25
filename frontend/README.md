# 🌿 Project Frontend Dashboard

Interactive Streamlit Web Dashboard for:
**"Deep Learning-Based Automated Bird Species Identification Using Acoustic Signals"**

---

## 🚀 Quickstart

### 1. Install Requirements
From this folder (`frontend/`):
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
Run via Streamlit:
```bash
streamlit run app.py
```
Or simply double-click **`run_dashboard.bat`** on Windows.

Alternatively, from the repository root:
```bash
streamlit run frontend/app.py
```

The application will be accessible at:
- **Local:** `http://localhost:8501`

---

## 📑 Dashboard Structure

- **🌿 Page 1: Problem Statement & Preprocessing Pipeline**
  - Executive Problem Statement Card & 3 bioacoustic field challenges (Environmental Noise, Sparse Calls, Hardware Limits).
  - BirdCLEF 2024 subset metrics (2,250 audio files, 15 species, 150 files/species, 32 kHz).
  - Zero-Data-Leakage scientific splitting guarantee (parent recording level 70% train, 15% val, 15% test).
  - 5-step audio-to-log-Mel spectrogram transformation pipeline.
  - Activity-Aware RMS Energy Window Selection (+13.3% accuracy boost).

- **🧠 Page 2: Individual Architecture Deep-Dives**
  - Interactive dropdown selector for all 7 project models:
    1. Baseline Custom CNN (424k params, Scratch)
    2. Pretrained ResNet-18 (11.2M params, Transfer Learning)
    3. MobileNetV3 + Squeeze-and-Excitation (1.67M params, Attention)
    4. Bioacoustic CRNN (786k params, CNN + BiGRU + Temporal Attention)
    5. EfficientNet-B0 (4.3M params, Compound Scaling)
    6. ConvNeXt-Tiny (28.0M params, Modern 7x7 Depthwise CNN)
    7. BirdNET (Cornell Foundation Model Benchmark)
  - For each model: 6 top KPI metric cards, architectural design rationale, viva defense notes, training convergence curves, 15x15 holdout confusion matrix, per-species performance table, and high-resolution chart expansions.

- **📊 Page 3: Comprehensive Comparative Benchmark**
  - Interactive master leaderboard table ($N=338$ holdout test set).
  - 6 publication-ready comparative figures (performance ladder, macro ROC curves, parameter efficiency scatter, per-species difficulty heatmap, windowing ablation study, multi-metric radar).
  - Key engineering conclusions and Raspberry Pi 4 edge deployment analysis.
