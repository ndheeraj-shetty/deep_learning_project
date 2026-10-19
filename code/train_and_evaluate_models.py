"""
Automated Training, Evaluation & Benchmarking Pipeline
Models:
  1. EfficientNet-B0 (Inverted Bottleneck CNN + Compound Scaling)
  2. ConvNeXt-Tiny (Modernized Pure Depthwise CNN)
  3. CRNN (Bioacoustic CNN + Bi-Directional GRU + Temporal Pooling)

Zero Data Leakage: Strict recording-level train/val/test manifests.
Saves checkpoints, curves, confusion matrices, and detailed JSON reports.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import json
import time
import math
import random
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import librosa
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# --- Reproducibility ---
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_artifacts"
AUDIO_DIR = PROJECT_ROOT / "sample_dataset" / "audio"
MODELS_DIR = PROJECT_ROOT / "model_artifacts"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Using: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# Load configs and vocabulary
with open(DATA_DIR / "config" / "preprocessing_config.json") as f:
    CONFIG = json.load(f)

with open(DATA_DIR / "vocab" / "species_mapping.json") as f:
    VOCAB = json.load(f)

NUM_CLASSES = VOCAB["num_classes"]
IDX_TO_SPECIES = {int(k): v for k, v in VOCAB["idx_to_species"].items()}
SPECIES_METADATA = VOCAB["species_metadata"]
CLASS_NAMES = [f"{IDX_TO_SPECIES[i]} ({SPECIES_METADATA[IDX_TO_SPECIES[i]]['common_name']})" for i in range(NUM_CLASSES)]

SR = CONFIG["sample_rate"]
CLIP_DUR = CONFIG["clip_duration"]
NUM_SAMPLES = int(SR * CLIP_DUR)
N_MELS = CONFIG["n_mels"]
N_FFT = CONFIG["n_fft"]
HOP_LEN = CONFIG["hop_length"]
WIN_LEN = CONFIG["win_length"]
FMIN = CONFIG["f_min"]
FMAX = CONFIG["f_max"]
TOP_DB = CONFIG["top_db"]

# --- Spectrogram Computation ---
def audio_to_mel(y: np.ndarray) -> np.ndarray:
    # Normalize peak to 0.95
    peak = np.max(np.abs(y))
    if peak > 1e-8:
        y = (y / peak) * 0.95
    # Fix length to 160,000 samples
    if len(y) > NUM_SAMPLES:
        start = (len(y) - NUM_SAMPLES) // 2
        y = y[start : start + NUM_SAMPLES]
    elif len(y) < NUM_SAMPLES:
        repeats = int(np.ceil(NUM_SAMPLES / len(y)))
        y = np.tile(y, repeats)[:NUM_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)
    mel_db = np.nan_to_num(mel_db, nan=-TOP_DB, posinf=0.0, neginf=-TOP_DB)
    # Standardize
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_norm.astype(np.float32)

def precompute_split_cache(manifest_csv: Path, cache_file: Path):
    if cache_file.exists():
        print(f"[CACHE] Loading cached spectrograms from: {cache_file.name}")
        data = torch.load(cache_file, map_location="cpu", weights_only=False)
        return data["mels"], data["labels"], data["filenames"]

    df = pd.read_csv(manifest_csv)
    print(f"[CACHE] Precomputing {len(df)} spectrograms for {manifest_csv.name}...")
    mels_list = []
    labels_list = []
    filenames_list = []

    t0 = time.time()
    for idx, row in df.iterrows():
        fpath = AUDIO_DIR / row["filename"]
        try:
            y, file_sr = sf.read(str(fpath), dtype="float32")
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if file_sr != SR:
                y = librosa.resample(y, orig_sr=file_sr, target_sr=SR)
        except Exception as e:
            y = np.zeros(NUM_SAMPLES, dtype=np.float32)

        mel = audio_to_mel(y)
        mels_list.append(torch.from_numpy(mel).unsqueeze(0)) # Shape (1, 128, 313)
        labels_list.append(int(row["species_idx"]))
        filenames_list.append(row["filename"])

    mels_tensor = torch.stack(mels_list) # (N, 1, 128, 313)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    torch.save({"mels": mels_tensor, "labels": labels_tensor, "filenames": filenames_list}, cache_file)
    print(f"[CACHE] Cached {len(df)} samples in {time.time() - t0:.1f}s to {cache_file.name}")
    return mels_tensor, labels_tensor, filenames_list

# Precompute/load caches
train_mels, train_labels, train_files = precompute_split_cache(DATA_DIR / "manifests" / "train_manifest.csv", CACHE_DIR / "train_cache.pt")
val_mels, val_labels, val_files = precompute_split_cache(DATA_DIR / "manifests" / "val_manifest.csv", CACHE_DIR / "val_cache.pt")
test_mels, test_labels, test_files = precompute_split_cache(DATA_DIR / "manifests" / "test_manifest.csv", CACHE_DIR / "test_cache.pt")

# --- SpecAugment ---
def apply_spec_augment(spec: torch.Tensor, max_f=12, max_t=18) -> torch.Tensor:
    # spec: (1, 128, 313)
    spec_aug = spec.clone()
    _, f_bins, t_frames = spec_aug.shape
    fill_val = spec_aug.min()

    # 2 Frequency masks
    for _ in range(2):
        f = random.randint(1, max_f)
        f0 = random.randint(0, f_bins - f)
        spec_aug[:, f0 : f0 + f, :] = fill_val

    # 2 Time masks
    for _ in range(2):
        t = random.randint(1, max_t)
        t0 = random.randint(0, t_frames - t)
        spec_aug[:, :, t0 : t0 + t] = fill_val

    return spec_aug

# --- PyTorch Cached Dataset ---
class CachedBioacousticDataset(Dataset):
    def __init__(self, mels: torch.Tensor, labels: torch.Tensor, is_train: bool = False, in_channels: int = 3):
        self.mels = mels
        self.labels = labels
        self.is_train = is_train
        self.in_channels = in_channels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel = self.mels[idx] # (1, 128, 313)
        if self.is_train and random.random() > 0.4:
            mel = apply_spec_augment(mel)

        if self.in_channels == 3:
            mel = mel.repeat(3, 1, 1) # (3, 128, 313)

        return mel, self.labels[idx]

# --- Model Architectures ---

# 1. EfficientNet-B0 (Pretrained MBConv + Bioacoustic Head)
def build_efficientnet(num_classes=15):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_feat = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(in_feat, 256, bias=False),
        nn.BatchNorm1d(256),
        nn.SiLU(inplace=True),
        nn.Dropout(p=0.25),
        nn.Linear(256, num_classes)
    )
    return m

# 2. ConvNeXt-Tiny (Pretrained Modernized Pure Depthwise CNN)
def build_convnext(num_classes=15):
    m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    in_feat = m.classifier[2].in_features
    m.classifier[2] = nn.Sequential(
        nn.Linear(in_feat, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(p=0.30),
        nn.Linear(256, num_classes)
    )
    return m

# 3. CRNN (CNN + Bidirectional GRU + Attention Pooling)
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
        # 4 Conv blocks pooling mainly frequency to preserve temporal resolution
        self.b1 = CRNNBlock(in_channels, 32, pool_stride=(2, 2))  # (128, 313) -> (64, 156)
        self.b2 = CRNNBlock(32, 64, pool_stride=(2, 2))          # (64, 156)  -> (32, 78)
        self.b3 = CRNNBlock(64, 128, pool_stride=(2, 1))         # (32, 78)   -> (16, 78)
        self.b4 = CRNNBlock(128, 128, pool_stride=(4, 1))        # (16, 78)   -> (4, 78)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))          # (128, 1, 78)
        
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25
        )
        
        # Temporal attention mechanism
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
        # x: (B, C, 128, 313)
        h = self.b4(self.b3(self.b2(self.b1(x))))
        h = self.freq_pool(h).squeeze(2) # (B, 128, T)
        h = h.permute(0, 2, 1)          # (B, T, 128)
        
        gru_out, _ = self.gru(h)        # (B, T, hidden*2)
        attn_weights = torch.softmax(self.attn(gru_out), dim=1) # (B, T, 1)
        context = torch.sum(gru_out * attn_weights, dim=1)      # (B, hidden*2)
        
        return self.fc(context)

# --- Training and Evaluation Routines ---
def train_and_evaluate_model(
    model_key: str,
    model_name: str,
    build_fn,
    in_channels: int,
    epochs: int = 20,
    lr: float = 3e-4,
    batch_size: int = 32
):
    print("\n" + "=" * 85)
    print(f"[START] INITIATING TRAINING: {model_name} (key: {model_key})")
    print("=" * 85)
    
    save_dir = MODELS_DIR / model_key
    save_dir.mkdir(parents=True, exist_ok=True)
    
    train_ds = CachedBioacousticDataset(train_mels, train_labels, is_train=True, in_channels=in_channels)
    val_ds = CachedBioacousticDataset(val_mels, val_labels, is_train=False, in_channels=in_channels)
    test_ds = CachedBioacousticDataset(test_mels, test_labels, is_train=False, in_channels=in_channels)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    model = build_fn(num_classes=NUM_CLASSES).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    history = {
        "epoch": [],
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_f1": []
    }
    
    best_val_f1 = -1.0
    best_epoch = -1
    best_ckpt_path = save_dir / f"best_{model_key}.pt"
    
    t_start = time.time()
    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        # --- Train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_loss += loss.item() * len(y)
            train_correct += (out.argmax(1) == y).sum().item()
            train_total += len(y)
            
        scheduler.step()
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_val_preds, all_val_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * len(y)
                preds = out.argmax(1)
                val_correct += (preds == y).sum().item()
                val_total += len(y)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_targets.extend(y.cpu().numpy())
                
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_macro_f1 = f1_score(all_val_targets, all_val_preds, average="macro", zero_division=0)
        
        history["epoch"].append(ep)
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_f1"].append(round(val_macro_f1, 4))
        
        is_best = val_macro_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_macro_f1
            best_epoch = ep
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_macro_f1": best_val_f1,
                "model_name": model_name,
                "model_key": model_key,
                "in_channels": in_channels,
                "total_params": total_params
            }, best_ckpt_path)
            
        star = " * (Best)" if is_best else ""
        print(f"Epoch [{ep:02d}/{epochs:02d}] {time.time()-ep_t0:.1f}s | "
              f"Tr Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.2f}% F1: {val_macro_f1*100:5.2f}%{star}")
              
    print(f"[COMPLETE] Trained in {time.time()-t_start:.1f}s. Best Epoch: {best_epoch} with Val Macro F1: {best_val_f1*100:.2f}%")
    
    # Save training history JSON
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)
        
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss", color="crimson", lw=2)
    axes[0].set_title(f"{model_name} - Cross Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history["epoch"], [a * 100 for a in history["train_acc"]], label="Train Acc (%)", color="royalblue", lw=2)
    axes[1].plot(history["epoch"], [a * 100 for a in history["val_acc"]], label="Val Acc (%)", color="crimson", lw=2)
    axes[1].plot(history["epoch"], [f * 100 for f in history["val_f1"]], label="Val Macro F1 (%)", color="forestgreen", lw=2, linestyle="--")
    axes[1].axvline(best_epoch, color="gold", linestyle=":", label=f"Best Ep {best_epoch}")
    axes[1].set_title(f"{model_name} - Accuracy & Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=200)
    plt.close()
    
    # Load best weights for Holdout Evaluation
    ckpt = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # --- 1. Center-Crop Test Evaluation ---
    test_preds, test_targets, test_probs = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
            test_probs.append(probs)
            test_preds.extend(np.argmax(probs, axis=1))
            test_targets.extend(y.numpy())
            
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_probs = np.concatenate(test_probs, axis=0)
    
    test_acc = accuracy_score(test_targets, test_preds)
    test_macro_f1 = f1_score(test_targets, test_preds, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(test_targets, test_preds, average="weighted", zero_division=0)
    test_macro_p = precision_score(test_targets, test_preds, average="macro", zero_division=0)
    test_macro_r = recall_score(test_targets, test_preds, average="macro", zero_division=0)
    
    top3_center = sum(1 for i in range(len(test_targets)) if test_targets[i] in np.argsort(test_probs[i])[::-1][:3]) / len(test_targets)
    
    center_report = {
        "model_name": model_name,
        "model_key": model_key,
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "test_accuracy": round(test_acc, 4),
        "test_accuracy_top3": round(top3_center, 4),
        "test_macro_precision": round(test_macro_p, 4),
        "test_macro_recall": round(test_macro_r, 4),
        "test_macro_f1": round(test_macro_f1, 4),
        "test_weighted_f1": round(test_weighted_f1, 4),
        "classification_report": classification_report(test_targets, test_preds, target_names=CLASS_NAMES, output_dict=True)
    }
    with open(save_dir / "test_evaluation_report.json", "w") as f:
        json.dump(center_report, f, indent=4)
        
    # Plot test confusion matrix
    cm_test = confusion_matrix(test_targets, test_preds)
    plot_cm(cm_test, [IDX_TO_SPECIES[i] for i in range(NUM_CLASSES)], save_dir / "test_confusion_matrix.png", f"{model_name} (Center Crop)")
    
    # --- 2. Activity-Aware Multi-Window Ensemble Test Evaluation ---
    print(f"\n[EVAL] Running Top-3 Multi-Window Temporal Ensemble for {model_name}...")
    ens_preds, ens_probs = [], []
    test_df = pd.read_csv(DATA_DIR / "manifests" / "test_manifest.csv")
    
    for _, row in test_df.iterrows():
        p = AUDIO_DIR / row["filename"]
        try:
            y, file_sr = sf.read(str(p), dtype="float32")
            if y.ndim > 1: y = np.mean(y, axis=1)
            if file_sr != SR: y = librosa.resample(y, orig_sr=file_sr, target_sr=SR)
        except Exception:
            y = np.zeros(NUM_SAMPLES, dtype=np.float32)
            
        windows = get_top_k_energy_windows(y, sr=SR, duration=CLIP_DUR, k=3)
        tensors = []
        for w in windows:
            m = audio_to_mel(w)
            t = torch.from_numpy(m).unsqueeze(0)
            if in_channels == 3:
                t = t.repeat(3, 1, 1)
            tensors.append(t)
        batch = torch.stack(tensors).to(DEVICE)
        with torch.no_grad():
            avg_prob = torch.softmax(model(batch), dim=1).mean(dim=0).cpu().numpy()
            
        ens_probs.append(avg_prob)
        ens_preds.append(int(np.argmax(avg_prob)))
        
    ens_preds = np.array(ens_preds)
    ens_probs = np.array(ens_probs)
    
    ens_acc = accuracy_score(test_targets, ens_preds)
    ens_top3 = sum(1 for i in range(len(test_targets)) if test_targets[i] in np.argsort(ens_probs[i])[::-1][:3]) / len(test_targets)
    ens_macro_f1 = f1_score(test_targets, ens_preds, average="macro", zero_division=0)
    ens_weighted_f1 = f1_score(test_targets, ens_preds, average="weighted", zero_division=0)
    ens_macro_p = precision_score(test_targets, ens_preds, average="macro", zero_division=0)
    ens_macro_r = recall_score(test_targets, ens_preds, average="macro", zero_division=0)
    
    ens_report = {
        "model_name": f"{model_name} (Top-3 Multi-Window Ensemble)",
        "model_key": model_key,
        "best_epoch": best_epoch,
        "test_accuracy_top1": round(ens_acc, 4),
        "test_accuracy_top3": round(ens_top3, 4),
        "test_macro_precision": round(ens_macro_p, 4),
        "test_macro_recall": round(ens_macro_r, 4),
        "test_macro_f1": round(ens_macro_f1, 4),
        "test_weighted_f1": round(ens_weighted_f1, 4),
        "classification_report": classification_report(test_targets, ens_preds, target_names=CLASS_NAMES, output_dict=True)
    }
    with open(save_dir / "final_evaluation_report.json", "w") as f:
        json.dump(ens_report, f, indent=4)
        
    cm_ens = confusion_matrix(test_targets, ens_preds)
    plot_cm(cm_ens, [IDX_TO_SPECIES[i] for i in range(NUM_CLASSES)], save_dir / "final_confusion_matrix.png", f"{model_name} (Multi-Window Ensemble)")
    
    print(f"[RESULTS] {model_name}:")
    print(f"   Center-Crop  : Top-1 Acc: {test_acc*100:5.2f}% | Top-3 Acc: {top3_center*100:5.2f}% | Macro F1: {test_macro_f1*100:5.2f}%")
    print(f"   Multi-Window : Top-1 Acc: {ens_acc*100:5.2f}% | Top-3 Acc: {ens_top3*100:5.2f}% | Macro F1: {ens_macro_f1*100:5.2f}%")
    
    return {
        "model_name": model_name,
        "model_key": model_key,
        "total_params": total_params,
        "best_val_f1": best_val_f1,
        "center_acc": test_acc,
        "center_top3": top3_center,
        "center_f1": test_macro_f1,
        "ens_acc": ens_acc,
        "ens_top3": ens_top3,
        "ens_f1": ens_macro_f1
    }

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

def plot_cm(cm, labels, save_path, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title=f"Confusion Matrix: {title}",
        ylabel="True Species",
        xlabel="Predicted Species"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Annotate numbers
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

if __name__ == "__main__":
    results = []
    
    # 1. EfficientNet-B0
    res_eff = train_and_evaluate_model(
        model_key="efficientnet_b0",
        model_name="Bioacoustic EfficientNet-B0 (MBConv)",
        build_fn=build_efficientnet,
        in_channels=3,
        epochs=20,
        lr=3e-4,
        batch_size=32
    )
    results.append(res_eff)
    
    # 2. ConvNeXt-Tiny
    res_cnx = train_and_evaluate_model(
        model_key="convnext_tiny",
        model_name="Bioacoustic ConvNeXt-Tiny (Modernized CNN)",
        build_fn=build_convnext,
        in_channels=3,
        epochs=20,
        lr=2e-4,
        batch_size=32
    )
    results.append(res_cnx)
    
    # 3. CRNN
    res_crnn = train_and_evaluate_model(
        model_key="crnn",
        model_name="Bioacoustic CRNN (CNN + Bi-GRU + Attention)",
        build_fn=lambda num_classes: BioacousticCRNN(num_classes=num_classes, in_channels=1),
        in_channels=1,
        epochs=25,
        lr=5e-4,
        batch_size=32
    )
    results.append(res_crnn)
    
    print("\n" + "=" * 90)
    print("=== ALL PLANNED MODELS COMPLETED! CONSOLIDATED BENCHMARK SUMMARY ===")
    print("=" * 90)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(MODELS_DIR / "new_models_summary.csv", index=False)
    print("=" * 90 + "\n")
