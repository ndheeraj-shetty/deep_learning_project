"""
Interactive Bioacoustic Model Tester & Inference Engine
Evaluates models on the holdout test set and performs single-file birdsong audio prediction.

Supported Models:
    --model efficientnet  (Bioacoustic EfficientNet-B0 MBConv)
    --model convnext      (Bioacoustic ConvNeXt-Tiny Modernized CNN)
    --model crnn          (Bioacoustic CNN + Bi-GRU + Attention)
    --model mobilenet     (MobileNetV3-Small + SE Attention)
    --model custom_cnn    (Baseline Custom 4-Block CNN)

Usage Examples:
    # 1. Run final holdout test evaluation:
    python code/test_model.py --model efficientnet --mode evaluate
    python code/test_model.py --model convnext --mode evaluate
    python code/test_model.py --model crnn --mode evaluate

    # 2. Predict single audio file:
    python code/test_model.py --model efficientnet --mode predict --audio sample_dataset/audio/comkin1/XC110343.ogg
    python code/test_model.py --model convnext --mode predict --audio sample_dataset/audio/barswa/XC133802.ogg
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

import torch
import torch.nn as nn
import torchvision.models as models
import librosa
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

# Paths
VOCAB_PATH = Path("data_artifacts/vocab/species_mapping.json")
CONFIG_PATH = Path("data_artifacts/config/preprocessing_config.json")
TEST_MANIFEST_PATH = Path("data_artifacts/manifests/test_manifest.csv")
AUDIO_BASE = Path("sample_dataset/audio")

CKPT_PATHS = {
    "custom_cnn": Path("model_artifacts/custom_cnn/best_custom_cnn.pt"),
    "mobilenet": Path("model_artifacts/optimized_mobilenet/best_mobilenet_model.pt"),
    "efficientnet": Path("model_artifacts/efficientnet_b0/best_efficientnet_b0.pt"),
    "convnext": Path("model_artifacts/convnext_tiny/best_convnext_tiny.pt"),
    "crnn": Path("model_artifacts/crnn/best_crnn.pt")
}

# --- Architecture Definitions ---

class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x): return self.block(x)

class BioacousticCustomCNN(nn.Module):
    def __init__(self, num_classes=15, in_channels=1, dropout_rate=0.35):
        super().__init__()
        self.block1 = ConvBlock(in_channels, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.global_pool(self.block4(self.block3(self.block2(self.block1(x))))))

def build_mobilenet(num_classes=15):
    m = models.mobilenet_v3_small(weights=None)
    in_features = m.classifier[0].in_features
    m.classifier = nn.Sequential(
        nn.Linear(in_features, 256, bias=False),
        nn.BatchNorm1d(256),
        nn.Hardswish(),
        nn.Dropout(p=0.35),
        nn.Linear(256, num_classes)
    )
    return m

def build_efficientnet(num_classes=15):
    m = models.efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(in_features, 256, bias=False),
        nn.BatchNorm1d(256),
        nn.SiLU(inplace=True),
        nn.Dropout(p=0.25),
        nn.Linear(256, num_classes)
    )
    return m

def build_convnext(num_classes=15):
    m = models.convnext_tiny(weights=None)
    in_features = m.classifier[2].in_features
    m.classifier[2] = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(p=0.30),
        nn.Linear(256, num_classes)
    )
    return m

class CRNNBlock(nn.Module):
    def __init__(self, in_c, out_c, pool_stride=(2, 1)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_stride, stride=pool_stride)
        )
    def forward(self, x): return self.conv(x)

class BioacousticCRNN(nn.Module):
    def __init__(self, num_classes=15, in_channels=1, hidden_dim=128):
        super().__init__()
        self.b1 = CRNNBlock(in_channels, 32, pool_stride=(2, 2))
        self.b2 = CRNNBlock(32, 64, pool_stride=(2, 2))
        self.b3 = CRNNBlock(64, 128, pool_stride=(2, 1))
        self.b4 = CRNNBlock(128, 128, pool_stride=(4, 1))
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        h = self.b4(self.b3(self.b2(self.b1(x))))
        h = self.freq_pool(h).squeeze(2).permute(0, 2, 1)
        gru_out, _ = self.gru(h)
        attn_weights = torch.softmax(self.attn(gru_out), dim=1)
        context = torch.sum(gru_out * attn_weights, dim=1)
        return self.fc(context)

def load_pipeline(model_type="efficientnet"):
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"Species vocabulary not found at: {VOCAB_PATH}")
        
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    num_classes = vocab["num_classes"]
    idx_to_species = {int(k): v for k, v in vocab["idx_to_species"].items()}
    metadata = vocab["species_metadata"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_path = CKPT_PATHS.get(model_type)
    if not ckpt_path or not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for model '{model_type}' at: {ckpt_path}")
        
    if model_type == "efficientnet":
        model = build_efficientnet(num_classes=num_classes).to(device)
        model_name = "Bioacoustic EfficientNet-B0 (MBConv)"
        is_3channel = True
    elif model_type == "convnext":
        model = build_convnext(num_classes=num_classes).to(device)
        model_name = "Bioacoustic ConvNeXt-Tiny (Modernized CNN)"
        is_3channel = True
    elif model_type == "crnn":
        model = BioacousticCRNN(num_classes=num_classes, in_channels=1).to(device)
        model_name = "Bioacoustic CRNN (CNN + Bi-GRU + Attention)"
        is_3channel = False
    elif model_type == "mobilenet":
        model = build_mobilenet(num_classes=num_classes).to(device)
        model_name = "Bioacoustic MobileNetV3-Small (SE Attention)"
        is_3channel = True
    elif model_type == "custom_cnn":
        model = BioacousticCustomCNN(num_classes=num_classes).to(device)
        model_name = "Baseline Custom CNN"
        is_3channel = False
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device, idx_to_species, metadata, ckpt, model_name, is_3channel

def get_top_k_energy_windows(y: np.ndarray, sr=32000, duration=5.0, k=3):
    window = int(sr * duration)
    if len(y) <= window:
        repeats = int(np.ceil(window / max(len(y), 1)))
        return [np.tile(y, repeats)[:window]]
    hop = int(sr * 1.0)
    windows = []
    energies = []
    for start in range(0, len(y) - window + 1, hop):
        segment = y[start : start + window]
        rms = np.sqrt(np.mean(segment ** 2))
        windows.append(segment)
        energies.append(rms)
    top_indices = np.argsort(energies)[::-1][:k]
    return [windows[i] for i in top_indices]

def extract_mel_tensor(clip: np.ndarray, sr=32000, is_3channel=True) -> torch.Tensor:
    peak = np.max(np.abs(clip))
    if peak > 1e-8:
        clip = (clip / peak) * 0.95
    mel = librosa.feature.melspectrogram(
        y=clip, sr=sr, n_fft=2048, hop_length=512, win_length=2048,
        n_mels=128, fmin=50.0, fmax=14000.0, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    mel_db = np.nan_to_num(mel_db, nan=-80.0, posinf=0.0, neginf=-80.0)
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    
    if is_3channel:
        return torch.from_numpy(mel_norm).float().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    else:
        return torch.from_numpy(mel_norm).float().unsqueeze(0).unsqueeze(0)

def predict_single_audio(file_path: str, model_type="efficientnet", top_k=3):
    model, device, idx_to_species, metadata, ckpt, model_name, is_3channel = load_pipeline(model_type)
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
        
    y, sr = sf.read(str(file_path), dtype="float32")
    if y.ndim > 1: y = np.mean(y, axis=1)
    if sr != 32000: y = librosa.resample(y, orig_sr=sr, target_sr=32000)
    
    clips = get_top_k_energy_windows(y, sr=32000, duration=5.0, k=top_k)
    tensors = torch.cat([extract_mel_tensor(c, sr=32000, is_3channel=is_3channel) for c in clips], dim=0).to(device)
    
    with torch.no_grad():
        probs = torch.softmax(model(tensors), dim=1).mean(dim=0).cpu().numpy()
        
    top_indices = np.argsort(probs)[::-1][:top_k]
    
    print("\n" + "=" * 80)
    print(f"BIOACOUSTIC INFERENCE PREDICTION FOR: {file_path.name}")
    print("=" * 80)
    print(f"Active Architecture : {model_name}")
    print(f"Selected Checkpoint : Epoch {ckpt['epoch']}")
    print(f"Inference Mode      : Activity-Aware Top-{top_k} Window Temporal Ensemble")
    print(f"Audio Duration      : {len(y)/32000:.2f} seconds")
    print("-" * 80)
    print("TOP PREDICTED CANDIDATES:")
    print("-" * 80)
    for rank, idx in enumerate(top_indices, 1):
        sp_code = idx_to_species[idx]
        common = metadata[sp_code]["common_name"]
        scientific = metadata[sp_code]["scientific_name"]
        conf = probs[idx] * 100
        bar_len = int(conf // 3)
        bar = "#" * bar_len + "-" * (33 - bar_len)
        print(f"Rank {rank} | [{bar}] {conf:>5.2f}% | [{sp_code:<8}] {common} ({scientific})")
    print("=" * 80 + "\n")

def evaluate_test_set(model_type="efficientnet"):
    model, device, idx_to_species, metadata, ckpt, model_name, is_3channel = load_pipeline(model_type)
    test_df = pd.read_csv(TEST_MANIFEST_PATH)
    
    print("\n" + "=" * 80)
    print(f"EVALUATING ON HOLDOUT TEST SET ({model_name})")
    print(f"Inference Method: Activity-Aware Multi-Window Ensemble (N={len(test_df)} recordings)")
    print("=" * 80)
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    for _, row in test_df.iterrows():
        p = AUDIO_BASE / row["filename"]
        try:
            y, sr = sf.read(str(p), dtype="float32")
            if y.ndim > 1: y = np.mean(y, axis=1)
            if sr != 32000: y = librosa.resample(y, orig_sr=sr, target_sr=32000)
        except Exception:
            y = np.zeros(160000, dtype=np.float32)
            
        clips = get_top_k_energy_windows(y, sr=32000, duration=5.0, k=3)
        tensors = torch.cat([extract_mel_tensor(c, sr=32000, is_3channel=is_3channel) for c in clips], dim=0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensors), dim=1).mean(dim=0).cpu().numpy()
            pred = int(np.argmax(probs))
            
        all_preds.append(pred)
        all_targets.append(int(row["species_idx"]))
        all_probs.append(probs)
        
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    top3_hits = sum(1 for i in range(len(all_targets)) if all_targets[i] in np.argsort(all_probs[i])[::-1][:3])
    top3_acc = top3_hits / len(all_targets)
    
    acc = accuracy_score(all_targets, all_preds)
    macro_p = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_r = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    print(f"Evaluated Test Samples : {len(all_targets)}")
    print(f"Top-1 Test Accuracy    : {acc*100:.2f}%")
    print(f"Top-3 Test Accuracy    : {top3_acc*100:.2f}%")
    print(f"Macro Precision        : {macro_p:.4f} ({macro_p*100:.2f}%)")
    print(f"Macro Recall           : {macro_r:.4f} ({macro_r*100:.2f}%)")
    print(f"Macro F1 Score         : {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print(f"Weighted F1 Score      : {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Bioacoustic Bird Species Identification Models")
    parser.add_argument("--model", type=str,
                        choices=["efficientnet", "convnext", "crnn", "mobilenet", "custom_cnn"],
                        default="efficientnet",
                        help="Model architecture: 'efficientnet', 'convnext', 'crnn', 'mobilenet', 'custom_cnn'")
    parser.add_argument("--mode", type=str, choices=["evaluate", "predict"], default="evaluate",
                        help="Execution mode: 'evaluate' on test set, or 'predict' on single audio")
    parser.add_argument("--audio", type=str, default="sample_dataset/audio/comkin1/XC110343.ogg",
                        help="Path to audio file for single prediction mode")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top candidates to display")
    
    args = parser.parse_args()
    if args.mode == "evaluate":
        evaluate_test_set(model_type=args.model)
    elif args.mode == "predict":
        predict_single_audio(args.audio, model_type=args.model, top_k=args.top_k)
