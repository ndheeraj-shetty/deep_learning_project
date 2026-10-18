"""
End-to-End Fine-Tuning of Bioacoustic Attention Network (MobileNetV3-Small + SE Attention)
With SpecAugment (Frequency & Time Masking), Label Smoothing, and Cosine Annealing.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)

CACHE_DIR = Path("data_artifacts/cache")
MANIFEST_DIR = Path("data_artifacts/manifests")
VOCAB_PATH = Path("data_artifacts/vocab/species_mapping.json")
OUTPUT_DIR = Path("model_artifacts/optimized_mobilenet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(MANIFEST_DIR / "train_manifest.csv")
val_df = pd.read_csv(MANIFEST_DIR / "val_manifest.csv")
test_df = pd.read_csv(MANIFEST_DIR / "test_manifest.csv")

with open(VOCAB_PATH) as f:
    vocab_payload = json.load(f)

species_to_idx = vocab_payload["species_to_idx"]
idx_to_species = {int(k): v for k, v in vocab_payload["idx_to_species"].items()}
num_classes = len(species_to_idx)

# SpecAugment Implementation
def apply_spec_augment(mel: np.ndarray, max_mask_pct=0.15, n_freq_masks=2, n_time_masks=2):
    """Applies frequency and time masking to log-Mel spectrogram."""
    mel = mel.copy()
    n_mels, n_steps = mel.shape
    
    # Frequency masking
    for _ in range(n_freq_masks):
        f = int(np.random.uniform(0, max_mask_pct) * n_mels)
        f0 = int(np.random.uniform(0, n_mels - f))
        mel[f0 : f0 + f, :] = mel.mean()
        
    # Time masking
    for _ in range(n_time_masks):
        t = int(np.random.uniform(0, max_mask_pct) * n_steps)
        t0 = int(np.random.uniform(0, n_steps - t))
        mel[:, t0 : t0 + t] = mel.mean()
        
    return mel

class CachedAudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cache_dir: Path, is_train: bool = False):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_path = self.cache_dir / (row['filename'].replace(".ogg", ".npy"))
        mel = np.load(npy_path)

        if self.is_train and np.random.rand() > 0.3:
            mel = apply_spec_augment(mel)

        # Standardize
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        
        # 3-channel visual representation for ImageNet backbone
        tensor_3c = torch.from_numpy(mel).float().unsqueeze(0).repeat(3, 1, 1)
        label = int(row['species_idx'])
        return tensor_3c, label

# Dataloaders
train_dataset = CachedAudioDataset(train_df, CACHE_DIR, is_train=True)
val_dataset = CachedAudioDataset(val_df, CACHE_DIR, is_train=False)
test_dataset = CachedAudioDataset(test_df, CACHE_DIR, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Build Model: MobileNetV3-Small with SE-Attention
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

# Replace classifier head with Bioacoustic Attention Head
in_features = model.classifier[0].in_features # 576
model.classifier = nn.Sequential(
    nn.Linear(in_features, 256, bias=False),
    nn.BatchNorm1d(256),
    nn.Hardswish(),
    nn.Dropout(p=0.35),
    nn.Linear(256, num_classes)
)
model = model.to(device)

# Differential learning rates: lower for backbone, higher for new head
params_backbone = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
params_head = [p for n, p in model.named_parameters() if n.startswith("classifier")]

optimizer = torch.optim.AdamW([
    {"params": params_backbone, "lr": 3e-4, "weight_decay": 1e-4},
    {"params": params_head, "lr": 1.5e-3, "weight_decay": 1e-3}
])
criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

print("\n" + "=" * 80)
print("TRAINING BIOACOUSTIC MOBILENETV3-SMALL (SE-ATTENTION) END-TO-END")
print("=" * 80)
print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Train Acc':<10} | {'Val Acc':<10} | {'Val Macro F1':<12}")
print("=" * 80)

best_val_f1 = -1.0
best_epoch = 0
best_weights = None
history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

for epoch in range(1, 21):
    t0 = time.time()
    model.train()
    running_loss = 0.0
    t_preds, t_targets = [], []
    for x_b, y_b in train_loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(x_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * len(y_b)
        t_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
        t_targets.extend(y_b.cpu().numpy())
        
    scheduler.step()
    train_loss = running_loss / len(train_dataset)
    train_acc = accuracy_score(t_targets, t_preds)
    
    # Validation
    model.eval()
    val_loss = 0.0
    v_preds, v_targets = [], []
    with torch.no_grad():
        for x_b, y_b in val_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            logits = model(x_b)
            loss = criterion(logits, y_b)
            val_loss += loss.item() * len(y_b)
            v_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            v_targets.extend(y_b.cpu().numpy())
            
    val_loss /= len(val_dataset)
    val_acc = accuracy_score(v_targets, v_preds)
    val_f1 = f1_score(v_targets, v_preds, average='macro', zero_division=0)
    
    history["epoch"].append(epoch)
    history["train_loss"].append(round(train_loss, 4))
    history["val_loss"].append(round(val_loss, 4))
    history["train_acc"].append(round(train_acc, 4))
    history["val_acc"].append(round(val_acc, 4))
    history["val_f1"].append(round(val_f1, 4))
    
    is_best = val_f1 > best_val_f1
    marker = "  [BEST SAVE]" if is_best else ""
    if is_best:
        best_val_f1 = val_f1
        best_epoch = epoch
        best_weights = model.state_dict().copy()
        
    dt = time.time() - t0
    print(f"{epoch:>2}/20  | {train_loss:>10.4f} | {val_loss:>10.4f} | {train_acc*100:>9.2f}% | {val_acc*100:>9.2f}% | {val_f1:>12.4f}{marker}  ({dt:.1f}s)")

# Load best weights
model.load_state_dict(best_weights)
torch.save({
    "epoch": best_epoch,
    "model_state_dict": model.state_dict(),
    "best_val_f1": best_val_f1,
    "architecture": "MobileNetV3_Small_SE_Attention"
}, OUTPUT_DIR / "best_mobilenet_model.pt")

# Evaluate on Holdout Test Set
print("\n" + "=" * 80)
print(f"EVALUATING ON HOLDOUT TEST SET (N={len(test_dataset)})")
print("=" * 80)

model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for x_b, y_b in test_loader:
        x_b = x_b.to(device)
        logits = model(x_b)
        test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        test_targets.extend(y_b.numpy())

test_acc = accuracy_score(test_targets, test_preds)
test_macro_f1 = f1_score(test_targets, test_preds, average='macro', zero_division=0)
test_weighted_f1 = f1_score(test_targets, test_preds, average='weighted', zero_division=0)

print(f"Holdout Test Accuracy   : {test_acc*100:.2f}%")
print(f"Holdout Test Macro F1   : {test_macro_f1:.4f} ({test_macro_f1*100:.2f}%)")
print(f"Holdout Test Weighted F1: {test_weighted_f1:.4f} ({test_weighted_f1*100:.2f}%)")
print("=" * 80)

target_names = [f"{idx_to_species[i]} ({vocab_payload['species_metadata'][idx_to_species[i]]['common_name']})" for i in range(num_classes)]
cls_report = classification_report(test_targets, test_preds, target_names=target_names, digits=4, zero_division=0)
print(cls_report)

# Save Evaluation Report JSON
eval_report = {
    "model_name": "Bioacoustic MobileNetV3-Small (Squeeze-and-Excitation Attention)",
    "test_accuracy": round(test_acc, 4),
    "test_macro_f1": round(test_macro_f1, 4),
    "test_weighted_f1": round(test_weighted_f1, 4),
    "best_epoch": best_epoch,
    "best_val_macro_f1": round(best_val_f1, 4),
    "classification_report": classification_report(test_targets, test_preds, target_names=target_names, output_dict=True, zero_division=0)
}
with open(OUTPUT_DIR / "test_evaluation_report.json", "w") as f:
    json.dump(eval_report, f, indent=4)

# Plot and Save Confusion Matrix
cm = confusion_matrix(test_targets, test_preds)
short_names = [idx_to_species[i] for i in range(num_classes)]

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=short_names, yticklabels=short_names, square=True)
plt.title(f"Optimized Attention Model Test Confusion Matrix (Accuracy: {test_acc*100:.2f}%)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Predicted Species Code", fontsize=12)
plt.ylabel("True Species Code", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "test_confusion_matrix.png", dpi=150)
plt.close()

# Plot and Save Learning Curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
ep_r = history["epoch"]
axes[0].plot(ep_r, history["train_loss"], label="Train Loss", color='#2980b9', lw=2)
axes[0].plot(ep_r, history["val_loss"], label="Val Loss", color='#e74c3c', lw=2)
axes[0].axvline(best_epoch, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
axes[0].set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(ep_r, [a*100 for a in history["train_acc"]], label="Train Acc (%)", color='#2980b9', lw=2)
axes[1].plot(ep_r, [a*100 for a in history["val_acc"]], label="Val Acc (%)", color='#e67e22', lw=2)
axes[1].plot(ep_r, [f*100 for f in history["val_f1"]], label="Val Macro F1 (%)", color='#27ae60', lw=2)
axes[1].axvline(best_epoch, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
axes[1].set_title("Accuracy & Macro F1 (%)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Metric (%)")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
plt.close()
print("All optimized artifacts successfully saved to:", OUTPUT_DIR)
