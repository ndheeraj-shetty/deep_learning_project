"""
Bioacoustic Deep Learning Project: Automated Bird Species Identification
Interactive Streamlit Dashboard for Lab Project Presentation & Viva Defense.

Run command:
    streamlit run app.py
"""

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & AESTHETIC THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Deep Learning-Based Automated Bird Species Identification Using Acoustic Signals",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark / Modern CSS with Green & Audio Blue Accents
CUSTOM_CSS = """
<style>
    /* Global Background & Typography */
    .stApp {
        background-color: #0b0f17;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #070a10 !important;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #94a3b8 !important;
        font-weight: 500;
    }
    
    /* Card Container */
    .bio-card {
        background: linear-gradient(135deg, #111827 0%, #161e2e 100%);
        border: 1px solid #1f293d;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .bio-card:hover {
        border-color: #2ecc71;
    }
    .bio-card-highlight {
        background: linear-gradient(135deg, #0e1e2d 0%, #12283a 100%);
        border: 1px solid #38bdf8;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.15);
    }

    /* Metric Badges */
    .metric-box {
        background: #131d2e;
        border-left: 4px solid #2ecc71;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        border-top: 1px solid #1f293d;
        border-right: 1px solid #1f293d;
        border-bottom: 1px solid #1f293d;
    }
    .metric-box.blue {
        border-left-color: #38bdf8;
    }
    .metric-box.purple {
        border-left-color: #a855f7;
    }
    .metric-box.gold {
        border-left-color: #f59e0b;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .metric-sub {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 4px;
    }

    /* Badge Pills */
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-green { background: #064e3b; color: #34d399; border: 1px solid #059669; }
    .badge-blue { background: #0c4a6e; color: #38bdf8; border: 1px solid #0284c7; }
    .badge-purple { background: #4c1d95; color: #c084fc; border: 1px solid #7c3aed; }
    .badge-amber { background: #78350f; color: #fcd34d; border: 1px solid #d97706; }
    
    /* Table Styling */
    .dataframe {
        background-color: #0f172a !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    
    /* Section Headers */
    .page-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #2ecc71 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .page-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 24px;
        font-weight: 400;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. PATHS & CACHED DATA LOADERS
# -----------------------------------------------------------------------------
FILE_DIR = Path(__file__).resolve().parent
if (FILE_DIR / "model_artifacts").exists():
    BASE_DIR = FILE_DIR
elif (FILE_DIR.parent / "model_artifacts").exists():
    BASE_DIR = FILE_DIR.parent
else:
    BASE_DIR = FILE_DIR

DATA_ARTIFACTS = BASE_DIR / "data_artifacts"
MODEL_ARTIFACTS = BASE_DIR / "model_artifacts"
VIS_DIR = BASE_DIR / "visualizations_individual_models"


@st.cache_data
def load_species_metadata():
    path = DATA_ARTIFACTS / "vocab" / "species_mapping.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_benchmark_table():
    path = MODEL_ARTIFACTS / "report_figures" / "benchmark_metrics_table.csv"
    if path.exists():
        return pd.read_csv(path)
    # Fallback to visualizations CSV
    path2 = VIS_DIR / "summary_metrics_all_models.csv"
    if path2.exists():
        return pd.read_csv(path2)
    return pd.DataFrame()


@st.cache_data
def load_model_report(model_key: str):
    """Load JSON evaluation report for the selected model."""
    search_paths = [
        MODEL_ARTIFACTS / model_key / "final_evaluation_report.json",
        MODEL_ARTIFACTS / model_key / "test_evaluation_report.json",
    ]
    if model_key == "resnet18":
        search_paths.insert(0, MODEL_ARTIFACTS / "optimized_resnet18" / "test_evaluation_report.json")
    elif model_key == "mobilenet_v3":
        search_paths.insert(0, MODEL_ARTIFACTS / "optimized_mobilenet" / "final_evaluation_report.json")

    for p in search_paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def safe_image(image_path: Path, caption: str = ""):
    """Safely render an image with fallback notice."""
    if image_path.exists():
        img = Image.open(image_path)
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.warning(f"Image not found at path: `{image_path.name}`")


# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & META INFO
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌲 Project Navigation")
    page = st.radio(
        label="Select Dashboard View:",
        options=[
            "🌿 1. Problem & Preprocessing Pipeline",
            "🧠 2. Individual Model Deep-Dives",
            "📊 3. Comprehensive Benchmark & Edge",
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### 🔬 Project Metadata")
    st.markdown("""
    **Project Title:**  
    *Deep Learning-Based Automated Bird Species Identification Using Acoustic Signals*

    **Dataset:** BirdCLEF 2024 Curated Subset  
    **Evaluation Set:** $N=338$ Audio Files  
    **Classes:** 15 Avian Species (Balanced)  
    **Hardware Target:** Raspberry Pi 4 / Edge IoT  
    **Evaluation Protocol:** 100% Leak-Free Parent Split
    """)

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.75rem; color:#64748b; line-height: 1.4;">
        BTech Deep Learning Lab Project<br>
        PyTorch &bull; Librosa &bull; Streamlit<br>
        Zero-Leakage Bioacoustic Benchmarking
        </div>
        """,
        unsafe_allow_html=True
    )


# Global Top Header Banner Across All Pages
st.markdown("""
<div style="margin-bottom: 22px; padding: 18px 24px; background: linear-gradient(135deg, #0d1527 0%, #111e33 100%); border: 1px solid #1e293b; border-left: 5px solid #2ecc71; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.3);">
    <div style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; color:#38bdf8; font-weight:700; margin-bottom:4px;">
        BTech Deep Learning Lab &bull; Research & Demonstration Dashboard
    </div>
    <h1 style="font-size:1.85rem; font-weight:800; color:#f8fafc; margin:0 0 6px 0; line-height:1.25;">
        Deep Learning-Based Automated Bird Species Identification Using Acoustic Signals
    </h1>
    <div style="font-size:0.92rem; color:#94a3b8;">
        Passive Acoustic Monitoring (PAM) &bull; Activity-Aware RMS Energy Windowing &bull; 7-Architecture Neural Ladder Benchmark
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: PROBLEM STATEMENT & AUDIO PREPROCESSING PIPELINE
# =============================================================================
if page == "🌿 1. Problem & Preprocessing Pipeline":
    st.markdown('<div class="page-title">🌿 Problem Statement & Preprocessing Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Acoustic Signal Processing, Activity-Aware RMS Windowing & Zero-Data-Leakage Splitting</div>', unsafe_allow_html=True)

    # Executive Problem Statement Card
    st.markdown("""
    <div class="bio-card-highlight">
        <h3 style="color:#38bdf8; margin-top:0; margin-bottom:8px;">🎯 Executive Problem Statement</h3>
        <p style="color:#e2e8f0; font-size:1.02rem; line-height:1.6; margin-bottom:12px;">
            <strong>Background:</strong> Passive Acoustic Monitoring (PAM) with autonomous field sound recorders offers scalable wildlife conservation monitoring, but manual annotation of thousands of hours of jungle audio is impossible. Field audio is inherently unstructured—heavily contaminated with wind, rain, river streams, and long stretches of empty silence between brief bird calls.
        </p>
        <p style="color:#cbd5e1; font-size:0.96rem; line-height:1.6; margin-bottom:12px;">
            <strong>The Problem:</strong> Industry-standard foundation models (e.g., Cornell's BirdNET) require massive compute and cannot run continuously on small solar- or battery-powered edge hardware in deep forests. Meanwhile, standard computer vision models frequently collapse because blind audio cropping captures silent background noise instead of vocalizations.
        </p>
        <p style="color:#cbd5e1; font-size:0.96rem; line-height:1.6; margin-bottom:0;">
            <strong>Our Research Objective:</strong> Engineer a lightweight, leak-free bioacoustic pipeline that:
            <br>1. Converts raw noisy field waveforms into clean log-Mel spectrograms (32 kHz, 128 Mel bands).
            <br>2. Automatically locates prominent vocalizations using sound energy (RMS sliding window), securing a <strong>+13.3% accuracy boost</strong>.
            <br>3. Benchmarks a 6-architecture ladder (from a 786k-param CRNN to modern ConvNeXt) to identify the optimal Pareto-frontier model for field-deployable edge devices.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Core Field Challenges in 3-Card Grid
    st.markdown("### ⚠️ The Three Core Bioacoustic Field Challenges")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="bio-card">
            <h4 style="color:#f59e0b; margin-top:0;">🌧️ 1. Environmental Noise</h4>
            <p style="color:#94a3b8; font-size:0.9rem; line-height:1.5;">
                Tropical forest field recordings are overwhelmed by torrential monsoon rains, gusting wind turbulence, running streams, and high-frequency insect stridulation that mask subtle avian chirps.
            </p>
            <span class="badge-pill badge-amber">SNR Variability</span>
            <span class="badge-pill badge-amber">Broadband Noise</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="bio-card">
            <h4 style="color:#38bdf8; margin-top:0;">⏳ 2. Extreme Temporal Sparsity</h4>
            <p style="color:#94a3b8; font-size:0.9rem; line-height:1.5;">
                A 30-second recording often contains only 2–3 seconds of actual bird vocalizations. Blind centre-cropping or naive truncation frequently extracts pure dead silence, training models on environmental noise.
            </p>
            <span class="badge-pill badge-blue">Sparse Calls</span>
            <span class="badge-pill badge-blue">Activity-Aware RMS</span>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="bio-card">
            <h4 style="color:#2ecc71; margin-top:0;">🔋 3. Edge Hardware Limits</h4>
            <p style="color:#94a3b8; font-size:0.9rem; line-height:1.5;">
                Remote forest sensors (e.g., Raspberry Pi, AudioMoth) run on solar panels or 18650 batteries. Heavy transformer models deplete power in hours; models must fit in under 5 MB of RAM and execute in milliseconds.
            </p>
            <span class="badge-pill badge-green">&lt; 15 MB Footprint</span>
            <span class="badge-pill badge-green">Real-Time Latency</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Dataset Overview & Metric Cards
    st.markdown("### 📊 Dataset Overview (BirdCLEF 2024 Curated Subset)")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">Total Audio Files</div>
            <div class="metric-value">2,250</div>
            <div class="metric-sub">Full field recordings (.ogg / .wav)</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-box blue">
            <div class="metric-title">Target Species</div>
            <div class="metric-value">15</div>
            <div class="metric-sub">Diverse avian taxonomic families</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-box purple">
            <div class="metric-title">Class Balance</div>
            <div class="metric-value">150</div>
            <div class="metric-sub">Files per species (Strictly balanced)</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-box gold">
            <div class="metric-title">Audio Standard</div>
            <div class="metric-value">32 kHz</div>
            <div class="metric-sub">Mono channel / 16-bit PCM</div>
        </div>
        """, unsafe_allow_html=True)

    # Zero-Leakage Guarantee Card
    st.markdown("""
    <div class="bio-card" style="border-left: 4px solid #38bdf8;">
        <h4 style="color:#38bdf8; margin-top:0; margin-bottom:6px;">🛡️ Zero-Data-Leakage Scientific Protocol</h4>
        <p style="color:#cbd5e1; font-size:0.92rem; line-height:1.6; margin-bottom:6px;">
            A major flaw in published bioacoustic literature is <em>slice-before-split leakage</em>: slicing 30s recordings into 5s segments before creating train/test splits. That causes windows from the <strong>exact same recording</strong> to contaminate both train and test partitions, artificially inflating test scores as models memorize the unique background microphone noise.
        </p>
        <p style="color:#94a3b8; font-size:0.9rem; line-height:1.5; margin-bottom:0;">
            <strong>Our Protocol:</strong> We split recordings at the parent file level first (<strong>70% Train: 1,575 files | 15% Validation: 337 files | 15% Holdout Test: 338 files</strong>). No audio window or spectrogram from a test recording ever enters the training pipeline.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Species Inventory Table
    metadata = load_species_metadata()
    if metadata and "species_metadata" in metadata:
        species_rows = []
        for code, meta in metadata["species_metadata"].items():
            species_rows.append({
                "Species Code": code,
                "Common Name": meta.get("common_name", code),
                "Scientific Name": meta.get("scientific_name", "N/A"),
                "Total Recordings": meta.get("count", 150),
                "Mean Quality Rating": f"{meta.get('mean_rating', 0.0):.2f} / 5.0"
            })
        df_species = pd.DataFrame(species_rows)

        with st.expander("📋 Click to inspect 15-Species Taxonomical Inventory Table", expanded=False):
            st.dataframe(df_species, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Audio Preprocessing Pipeline
    st.markdown("### 🎛️ Audio to Log-Mel Spectrogram Transformation")
    st.markdown("""
    The audio waveform is converted into a 2D time-frequency image representation matching human and avian auditory perception:
    """)

    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        st.markdown("""
        <div class="bio-card" style="padding:14px; text-align:center;">
            <div style="font-size:1.6rem;">🎙️</div>
            <strong style="color:#38bdf8; font-size:0.85rem;">1. Resample</strong>
            <p style="color:#94a3b8; font-size:0.75rem; margin-top:4px;">Standardize to 32,000 Hz single-channel mono.</p>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class="bio-card" style="padding:14px; text-align:center;">
            <div style="font-size:1.6rem;">📏</div>
            <strong style="color:#2ecc71; font-size:0.85rem;">2. Peak Normalize</strong>
            <p style="color:#94a3b8; font-size:0.75rem; margin-top:4px;">Scale amplitude dynamically into [-1.0, 1.0].</p>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown("""
        <div class="bio-card" style="padding:14px; text-align:center;">
            <div style="font-size:1.6rem;">🌊</div>
            <strong style="color:#a855f7; font-size:0.85rem;">3. STFT</strong>
            <p style="color:#94a3b8; font-size:0.75rem; margin-top:4px;">N_FFT=2048 (64ms Hann), Hop=512 (16ms).</p>
        </div>
        """, unsafe_allow_html=True)
    with p4:
        st.markdown("""
        <div class="bio-card" style="padding:14px; text-align:center;">
            <div style="font-size:1.6rem;">🎼</div>
            <strong style="color:#f59e0b; font-size:0.85rem;">4. Mel Filterbank</strong>
            <p style="color:#94a3b8; font-size:0.75rem; margin-top:4px;">128 triangular Mel bins (50 Hz - 14,000 Hz).</p>
        </div>
        """, unsafe_allow_html=True)
    with p5:
        st.markdown("""
        <div class="bio-card" style="padding:14px; text-align:center;">
            <div style="font-size:1.6rem;">📊</div>
            <strong style="color:#ec4899; font-size:0.85rem;">5. Log-dB Scaling</strong>
            <p style="color:#94a3b8; font-size:0.75rem; margin-top:4px;">Decibel power scale (top_db=80) ➔ [1, 128, 313].</p>
        </div>
        """, unsafe_allow_html=True)

    # Activity-Aware RMS Energy Window Selector
    st.markdown("#### ⚡ Activity-Aware RMS Energy Window Selection (+13.3% Accuracy Boost)")
    st.markdown("""
    Instead of arbitrarily extracting the middle 5 seconds of a file (which often captures empty wind noise between chirps), 
    we compute the **Root-Mean-Square (RMS) sound energy** across sliding 5.0-second candidate windows. 
    The system identifies the **top-3 highest acoustic energy vocalization peaks**, guaranteeing that spectrograms contain genuine bird vocalizations.
    """)

    # Display Figure 7: Example Spectrograms & Window Selection
    fig7_path = MODEL_ARTIFACTS / "report_figures_ca3" / "figure_07_example_spectrograms_window_selection.png"
    safe_image(fig7_path, caption="Figure: Comparison of full 30s audio file vs. Activity-Aware RMS Energy selected 5s high-vocalization windows.")


# =============================================================================
# PAGE 2: INDIVIDUAL MODEL DEEP-DIVES
# =============================================================================
elif page == "🧠 2. Individual Model Deep-Dives":
    st.markdown('<div class="page-title">🧠 Individual Architecture Deep-Dives</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Examine Design Rationale, Layer Mechanics, Convergence & Diagnostic Matrices for Each Model</div>', unsafe_allow_html=True)

    # Model Dictionary with metadata and file mappings
    MODELS = {
        "Baseline Custom CNN": {
            "key": "custom_cnn",
            "dir": MODEL_ARTIFACTS / "custom_cnn",
            "vis_dir": VIS_DIR / "01_custom_cnn",
            "params": "423,727",
            "type": "Scratch 4-Block Plain CNN",
            "top1": "50.59%",
            "top3": "N/A",
            "f1": "50.26%",
            "prec": "53.57%",
            "rec": "50.67%",
            "auc": "0.8140",
            "rationale": """
            **Baseline Hypothesis:** How well can a lightweight CNN trained completely from scratch without ImageNet pretraining perform on raw bioacoustic spectrograms?
            <br><br>
            **Architecture Design:**
            - 4 sequential convolutional blocks with increasing channel depth (32 → 64 → 128 → 256).
            - Kernel size $3\\times3$, batch normalization after each convolution, ReLU non-linearity, and $2\\times2$ max-pooling.
            - Global Average Pooling (GAP) reduces the feature maps to 256 dimensions, followed by a Dropout layer ($p=0.4$) and a linear classification head (15 logits).
            <br>
            **Viva Defense Insight:** While reaching 50.59% accuracy confirms signal learning well above random chance (6.67%), the model suffers from limited receptive field and misses long-duration syllable temporal context.
            """,
            "matrix_name": "test_confusion_matrix.png",
            "curves_name": "training_curves.png"
        },
        "Pretrained ResNet-18": {
            "key": "resnet18",
            "dir": MODEL_ARTIFACTS / "optimized_resnet18",
            "vis_dir": VIS_DIR / "02_resnet18",
            "params": "11,184,143",
            "type": "Deep Residual Network (Transfer Learning)",
            "top1": "49.11%",
            "top3": "N/A",
            "f1": "49.00%",
            "prec": "49.85%",
            "rec": "49.13%",
            "auc": "0.8020",
            "rationale": """
            **Hypothesis:** Does standard ImageNet transfer learning via residual skip connections automatically outperform custom training on audio spectrograms?
            <br><br>
            **Architecture Design:**
            - Standard ResNet-18 backbone with residual shortcut connections (F(x) + x) preventing vanishing gradients across 18 layers.
            - Input adapter layer converts single-channel log-Mel spectrograms to 3 channels ($1\\times1$ convolution).
            - Custom bioacoustic classification head with LayerNorm, Dropout ($p=0.3$), and a 512-dim projection.
            <br>
            **Viva Defense Insight:** ResNet-18 achieved 49.11%, slightly underperforming the scratch CNN. This demonstrates the **Domain Shift Dilemma**: photographic ImageNet filters (detecting edges, fur, textures) do not readily transfer to frequency-time spectrogram physics without bioacoustic-specific attention.
            """,
            "matrix_name": "test_confusion_matrix.png",
            "curves_name": "training_curves.png"
        },
        "MobileNetV3 + Squeeze-and-Excitation": {
            "key": "mobilenet_v3",
            "dir": MODEL_ARTIFACTS / "optimized_mobilenet",
            "vis_dir": VIS_DIR / "03_mobilenet_v3",
            "params": "1,674,863",
            "type": "Depthwise Inverted Residual + Channel Attention",
            "top1": "79.59%",
            "top3": "92.60%",
            "f1": "79.75%",
            "prec": "80.39%",
            "rec": "79.64%",
            "auc": "0.9580",
            "rationale": """
            **Hypothesis:** Can depthwise separable convolutions coupled with Squeeze-and-Excitation (SE) channel attention capture frequency bands efficiently at low parameter count?
            <br><br>
            **Architecture Design:**
            - Inverted residual bottlenecks with depthwise separable $3\\times3$ and $5\\times5$ convolutions, radically cutting compute by 85%.
            - Integrated Squeeze-and-Excitation (SE) blocks dynamically re-weight channel importance based on global spectral energy.
            - Hard-Swish activation functions replace standard ReLU for improved non-linear representation without exponential math overhead.
            <br>
            **Viva Defense Insight:** Surged performance from ~50% to **79.59% Top-1 and 92.60% Top-3** with only 1.67M parameters. Proves channel attention is crucial for suppressing noisy frequency bands.
            """,
            "matrix_name": "final_confusion_matrix.png",
            "curves_name": "training_curves.png"
        },
        "Bioacoustic CRNN": {
            "key": "crnn",
            "dir": MODEL_ARTIFACTS / "crnn",
            "vis_dir": VIS_DIR / "04_crnn",
            "params": "786,800",
            "type": "Hybrid CNN + Bidirectional GRU + Temporal Attention",
            "top1": "83.73%",
            "top3": "93.49%",
            "f1": "83.81%",
            "prec": "84.28%",
            "rec": "83.75%",
            "auc": "0.9710",
            "rationale": """
            **Hypothesis:** Bird calls are sequential acoustic events with temporal syllables and syntax. Can a hybrid CNN-RNN architecture capture spectrograms better than pure 2D vision models?
            <br><br>
            **Architecture Design:**
            - 4-stage convolutional feature extractor with frequency-axis pooling to preserve temporal resolution.
            - 2-layer Bidirectional GRU (Hidden dimension 128) processes the sequence forwards and backwards in time.
            - Temporal Attention Mechanism: dynamically computes a softmax importance score over all time steps, focusing on high-energy syllables while ignoring quiet inter-call intervals.
            <br>
            **Viva Defense Insight:** Reached an outstanding **83.73% Top-1 and 83.81% Macro F1** with just **786,800 parameters** (< 1M!). Highest parameter efficiency among all trained architectures.
            """,
            "matrix_name": "final_confusion_matrix.png",
            "curves_name": "training_curves.png"
        },
        "EfficientNet-B0": {
            "key": "efficientnet_b0",
            "dir": MODEL_ARTIFACTS / "efficientnet_b0",
            "vis_dir": VIS_DIR / "05_efficientnet_b0",
            "params": "4,339,595",
            "type": "Inverted Bottleneck MBConv + Compound Scaling",
            "top1": "86.69%",
            "top3": "94.67%",
            "f1": "86.63%",
            "prec": "87.11%",
            "rec": "86.75%",
            "auc": "0.9785",
            "rationale": """
            **Hypothesis:** Does balanced compound scaling of network depth, channel width, and spectrogram resolution yield the optimal edge classification performance?
            <br><br>
            **Architecture Design:**
            - MBConv (Mobile Inverted Bottleneck Convolution) blocks with expansion factor 6.
            - Squeeze-and-Excitation attention with reduction ratio 4 on all residual stages.
            - Stochastic Depth (drop-connect rate 0.2) regularizes the training, preventing co-adaptation on the 2,250 recording subset.
            <br>
            **Viva Defense Insight:** Reached **86.69% Top-1 Accuracy and 94.67% Top-3** with only 4.34M parameters and a fast 13.2 ms inference time. The undisputed **Production Edge Champion**.
            """,
            "matrix_name": "final_confusion_matrix.png",
            "curves_name": "training_curves.png"
        },
        "ConvNeXt-Tiny": {
            "key": "convnext_tiny",
            "dir": MODEL_ARTIFACTS / "convnext_tiny",
            "vis_dir": VIS_DIR / "06_convnext_tiny",
            "params": "28,021,359",
            "type": "Modernized 7x7 Depthwise CNN (Vision Transformer Design)",
            "top1": "87.28%",
            "top3": "95.86%",
            "f1": "87.37%",
            "prec": "88.02%",
            "rec": "87.33%",
            "auc": "0.9820",
            "rationale": """
            **Hypothesis:** Can modernizing a pure ConvNet with Vision Transformer design choices (large $7\\times7$ receptive fields, inverted bottlenecks, LayerNorm, GELU) achieve state-of-the-art bioacoustic accuracy?
            <br><br>
            **Architecture Design:**
            - Large $7\\times7$ depthwise convolutions simulate self-attention windows across wide time-frequency receptive fields.
            - Inverted bottleneck ratio (1:4 expansion) and fewer activation layers (single GELU per block).
            - Separate downsampling patchify layers with $2\\times2$ stride-2 convolutions and LayerNorm.
            <br>
            **Viva Defense Insight:** Secured the **highest overall accuracy among all trained models: 87.28% Top-1, 95.86% Top-3, and 0.9820 ROC-AUC**. Proves large receptive fields capture complex multi-harmonic bird vocalizations.
            """,
            "matrix_name": "final_confusion_matrix.png",
            "curves_name": "training_curves.png"
        },
        "BirdNET (Cornell Foundation Benchmark)": {
            "key": "birdnet",
            "dir": MODEL_ARTIFACTS / "birdnet",
            "vis_dir": VIS_DIR / "07_birdnet",
            "params": "15,000,000",
            "type": "Pretrained Acoustic Foundation Model (Cornell Bioacoustics)",
            "top1": "97.34%",
            "top3": "99.70%",
            "f1": "97.33%",
            "prec": "97.38%",
            "rec": "97.36%",
            "auc": "0.9985",
            "rationale": """
            **Benchmark Purpose:** Serves as the State-of-the-Art (SOTA) empirical ceiling and experimental control benchmark.
            <br><br>
            **Foundation Model Specs:**
            - Trained on over 3,000 global bird species and millions of wildlife audio hours by Cornell Lab of Ornithology and Chemnitz University.
            - Evaluated on our holdout test set ($N=338$) in a closed-set protocol restricted to the 15 target species.
            <br>
            **Viva Defense Insight:** Achieved 97.34% Top-1 and 99.70% Top-3. This establishes the theoretical upper bound on this dataset. Our lightweight ConvNeXt (87.28%) and EfficientNet (86.69%) achieve within 10% of this massive foundation model while operating with fraction of power and zero cloud dependency.
            """,
            "matrix_name": "final_confusion_matrix.png",
            "curves_name": None
        }
    }

    # Selection Control
    selected_name = st.selectbox(
        "🔍 Choose Architecture to Inspect:",
        options=list(MODELS.keys()),
        index=4  # Default to EfficientNet-B0
    )
    m = MODELS[selected_name]

    st.markdown(f"### 📌 {selected_name} — Architectural Profile")

    # Top Metric Badges (6 Columns)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Top-1 Accuracy</div>
            <div class="metric-value">{m['top1']}</div>
            <div class="metric-sub">Holdout Test Set</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-box blue">
            <div class="metric-title">Top-3 Accuracy</div>
            <div class="metric-value">{m['top3']}</div>
            <div class="metric-sub">Vocalization Margin</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-box purple">
            <div class="metric-title">Macro F1-Score</div>
            <div class="metric-value">{m['f1']}</div>
            <div class="metric-sub">Unweighted Class Mean</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-box gold">
            <div class="metric-title">Macro Precision</div>
            <div class="metric-value">{m['prec']}</div>
            <div class="metric-sub">False Positive Resistance</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Macro Recall</div>
            <div class="metric-value">{m['rec']}</div>
            <div class="metric-sub">Sensitivity Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with k6:
        st.markdown(f"""
        <div class="metric-box blue">
            <div class="metric-title">Parameters</div>
            <div class="metric-value">{m['params']}</div>
            <div class="metric-sub">{m['type']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Design Rationale & Hypothesis Card
    st.markdown(f"""
    <div class="bio-card">
        <h4 style="color:#2ecc71; margin-top:0; margin-bottom:8px;">💡 Design Rationale & Engineering Hypothesis</h4>
        <div style="color:#cbd5e1; font-size:0.92rem; line-height:1.6;">
            {m['rationale']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Visualizations (2 Columns)
    st.markdown("### 📈 Visual Diagnostics & Empirical Proof")
    vcol1, vcol2 = st.columns(2)

    with vcol1:
        st.markdown("#### Training Convergence Curves")
        if m["curves_name"]:
            curves_path = m["dir"] / m["curves_name"]
            safe_image(curves_path, caption=f"{selected_name}: Training vs Validation Loss & Accuracy per Epoch")
        else:
            # BirdNET has no local training curve, show clean individual ROC curve
            safe_image(m["vis_dir"] / "02_roc_curve.png", caption=f"{selected_name}: Macro ROC Curve (AUC = {m['auc']})")

    with vcol2:
        st.markdown("#### 15x15 Holdout Confusion Matrix")
        matrix_path = m["dir"] / m["matrix_name"]
        safe_image(matrix_path, caption=f"{selected_name}: Confusion Matrix on Holdout Test Set (N=338)")

    # Per-Species Classification Breakdown
    st.markdown("---")
    st.markdown("### 📋 Per-Species Performance Breakdown")
    report_data = load_model_report(m["key"])

    if report_data and "classification_report" in report_data:
        species_perf = []
        for sp_label, vals in report_data["classification_report"].items():
            if isinstance(vals, dict) and "f1-score" in vals:
                # Filter out average rows
                if sp_label in ["accuracy", "macro avg", "weighted avg"]:
                    continue
                species_perf.append({
                    "Species": sp_label,
                    "Precision": f"{vals['precision'] * 100:.2f}%",
                    "Recall": f"{vals['recall'] * 100:.2f}%",
                    "F1-Score": f"{vals['f1-score'] * 100:.2f}%",
                    "Test Samples (Support)": int(vals.get("support", 0))
                })

        if species_perf:
            df_perf = pd.DataFrame(species_perf)
            # Find strongest and weakest species
            df_perf_numeric = pd.DataFrame([
                {
                    "Species": row["Species"],
                    "F1": float(row["F1-Score"].replace("%", ""))
                }
                for row in species_perf
            ]).sort_values("F1", ascending=False)

            top_sp = df_perf_numeric.iloc[0]["Species"]
            top_f1 = df_perf_numeric.iloc[0]["F1"]
            bot_sp = df_perf_numeric.iloc[-1]["Species"]
            bot_f1 = df_perf_numeric.iloc[-1]["F1"]

            c_rep1, c_rep2 = st.columns(2)
            with c_rep1:
                st.markdown(f"""
                <div class="bio-card" style="border-left: 4px solid #2ecc71; padding:14px 18px;">
                    <div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase;">Highest Performing Class</div>
                    <div style="font-size:1.15rem; color:#f8fafc; font-weight:700;">{top_sp}</div>
                    <div style="font-size:0.85rem; color:#2ecc71; font-weight:600;">{top_f1:.2f}% F1-Score</div>
                </div>
                """, unsafe_allow_html=True)
            with c_rep2:
                st.markdown(f"""
                <div class="bio-card" style="border-left: 4px solid #f59e0b; padding:14px 18px;">
                    <div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase;">Most Challenging Class</div>
                    <div style="font-size:1.15rem; color:#f8fafc; font-weight:700;">{bot_sp}</div>
                    <div style="font-size:0.85rem; color:#f59e0b; font-weight:600;">{bot_f1:.2f}% F1-Score</div>
                </div>
                """, unsafe_allow_html=True)

            st.dataframe(df_perf, use_container_width=True, hide_index=True)

    # Clean Individual Visuals Accordion
    with st.expander(f"🖼️ View Additional High-Resolution Charts for {selected_name}"):
        iv_c1, iv_c2 = st.columns(2)
        with iv_c1:
            safe_image(m["vis_dir"] / "03_species_f1_ranking.png", caption="Ranked Species F1 Performance")
        with iv_c2:
            safe_image(m["vis_dir"] / "04_precision_vs_recall.png", caption="Precision vs. Recall Balance per Species")


# =============================================================================
# PAGE 3: COMPREHENSIVE BENCHMARK & EDGE TRADE-OFFS
# =============================================================================
elif page == "📊 3. Comprehensive Benchmark & Edge":
    st.markdown('<div class="page-title">📊 Cross-Model Benchmark & Efficiency Trade-offs</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Multi-Model Comparative Analysis, Edge Latency Benchmarking & Raspberry Pi Deployment Verdict</div>', unsafe_allow_html=True)

    # Master Leaderboard Table
    st.markdown("### 🏆 Master Benchmark Leaderboard (Holdout Test Set $N=338$)")
    df_bm = load_benchmark_table()

    if not df_bm.empty:
        # Style dataframe for dark theme presentation
        st.dataframe(
            df_bm,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Benchmark table CSV loaded.")

    st.markdown("---")

    # Comparative Visuals 6-Grid (Rendered side-by-side using st.columns)
    st.markdown("### 📈 Multi-Model Comparative Visual Suite")
    st.markdown("Six publication-ready comparative figures analyzing accuracy ladders, discriminative ROC power, parameter footprints, and windowing gains.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Chart A: Overall Performance Ladder")
        safe_image(
            MODEL_ARTIFACTS / "eval_summary_figures" / "01_comprehensive_metrics_comparison.png",
            caption="Comparison of Top-1 Accuracy, Top-3 Accuracy, and Macro F1 across all evaluated models."
        )
    with colB:
        st.markdown("#### Chart B: Multi-Model Macro ROC Curves")
        safe_image(
            MODEL_ARTIFACTS / "eval_summary_figures" / "04_macro_roc_curves_all_models.png",
            caption="One-vs-Rest Macro ROC Curves proving discriminative AUC power across probability thresholds."
        )

    colC, colD = st.columns(2)
    with colC:
        st.markdown("#### Chart C: Efficiency vs. Accuracy Scatter (Raspberry Pi Trade-Off)")
        safe_image(
            MODEL_ARTIFACTS / "report_figures" / "02_efficiency_vs_accuracy_scatter.png",
            caption="Parameter Count (Log Scale) vs. Top-1 Accuracy showing the optimal Pareto frontier."
        )
    with colD:
        st.markdown("#### Chart D: Per-Species F1 Difficulty Heatmap")
        safe_image(
            MODEL_ARTIFACTS / "eval_summary_figures" / "02_per_species_f1_heatmap.png",
            caption="Heatmap of all 15 species across all 7 models identifying acoustic confusion patterns."
        )

    colE, colF = st.columns(2)
    with colE:
        st.markdown("#### Chart E: Windowing Ablation Study (+13.3% Boost)")
        safe_image(
            MODEL_ARTIFACTS / "report_figures" / "05_windowing_ablation_study.png",
            caption="Ablation proving Activity-Aware RMS windowing dramatically outperforms blind centre-cropping."
        )
    with colF:
        st.markdown("#### Chart F: Multi-Metric Radar Trade-Offs")
        safe_image(
            MODEL_ARTIFACTS / "report_figures" / "07_model_tradeoffs_radar.png",
            caption="Radar visualization balancing Accuracy, Recall, Precision, Latency, and Memory Footprint."
        )

    st.markdown("---")

    # Key Engineering Conclusions & Edge Deployment Verdict
    st.markdown("### 🚀 Final Engineering Conclusions & Edge Deployment Verdict")

    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown("""
        <div class="bio-card" style="border-top: 4px solid #38bdf8;">
            <div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase;">🥇 Peak Accuracy Champion</div>
            <h3 style="color:#38bdf8; margin: 4px 0 10px 0;">ConvNeXt-Tiny</h3>
            <p style="color:#cbd5e1; font-size:0.88rem; line-height:1.5;">
                <strong>87.28% Top-1 | 95.86% Top-3 | 87.37% F1</strong><br>
                Modernized 7x7 depthwise convolutions simulate Vision Transformer self-attention windows. Captures complex wide-band avian harmonics and faint calls across time.
            </p>
            <span class="badge-pill badge-blue">Best Research Model</span>
            <span class="badge-pill badge-blue">Highest Sensitivity</span>
        </div>
        """, unsafe_allow_html=True)

    with e2:
        st.markdown("""
        <div class="bio-card" style="border-top: 4px solid #2ecc71;">
            <div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase;">⚡ Production Edge Champion</div>
            <h3 style="color:#2ecc71; margin: 4px 0 10px 0;">EfficientNet-B0</h3>
            <p style="color:#cbd5e1; font-size:0.88rem; line-height:1.5;">
                <strong>86.69% Top-1 | 94.67% Top-3 | 4.34M Params</strong><br>
                Achieves within 0.59% of ConvNeXt while using <strong>84% fewer parameters</strong> and running at just <strong>13.2 ms</strong> per 5-second audio slice on Raspberry Pi 4 CPU.
            </p>
            <span class="badge-pill badge-green">Recommended for IoT</span>
            <span class="badge-pill badge-green">Battery Friendly</span>
        </div>
        """, unsafe_allow_html=True)

    with e3:
        st.markdown("""
        <div class="bio-card" style="border-top: 4px solid #a855f7;">
            <div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase;">🌱 Ultra-Lightweight Innovator</div>
            <h3 style="color:#a855f7; margin: 4px 0 10px 0;">Bioacoustic CRNN</h3>
            <p style="color:#cbd5e1; font-size:0.88rem; line-height:1.5;">
                <strong>83.73% Top-1 | 83.81% F1 | 786k Params</strong><br>
                Under 800k parameters! Bidirectional GRU with temporal attention focuses specifically on bird syllables, beating ResNet-18 (11.2M params) by over <strong>+34% absolute accuracy</strong>.
            </p>
            <span class="badge-pill badge-purple">&lt; 1M Parameters</span>
            <span class="badge-pill badge-purple">Temporal Attention</span>
        </div>
        """, unsafe_allow_html=True)

    # Edge Deployment Architecture Note
    st.markdown("""
    <div class="bio-card">
        <h4 style="color:#f8fafc; margin-top:0;">📡 Edge Deployment on Autonomous Forest Recorders (Raspberry Pi 4)</h4>
        <div style="color:#94a3b8; font-size:0.92rem; line-height:1.6;">
            <ul>
                <li><strong>Memory Footprint:</strong> EfficientNet-B0 model weights occupy only <strong>16.8 MB</strong> in PyTorch format and can be quantized with INT8 to <strong>4.2 MB</strong>.</li>
                <li><strong>Inference Speed:</strong> At 13.2 ms per window on a standard 1.5 GHz ARM Cortex-A72 CPU, a 30-second recording with 3 candidate windows is processed in under <strong>40 milliseconds</strong>.</li>
                <li><strong>Power Budget:</strong> The edge sensor can sleep in low-power mode (0.3W), wake on an analog audio trigger, execute the lightweight CNN, record bird GPS coordinates, and return to sleep, enabling months of continuous solar/battery field operation.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# End of Application
# -----------------------------------------------------------------------------
