"""
Fast Pretrained ResNet-18 Feature Extraction & Classifier Optimization.
Extracts 512-dim visual-acoustic embeddings from pretrained ResNet-18,
then trains an optimized deep classifier head with MixUp, Label Smoothing, and Cosine Annealing.
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
OUTPUT_DIR = Path("model_artifacts/optimized_resnet18")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(MANIFEST_DIR / "train_manifest.csv")
val_df = pd.read_csv(MANIFEST_DIR / "val_manifest.csv")
test_df = pd.read_csv(MANIFEST_DIR / "test_manifest.csv")

with open(VOCAB_PATH) as f:
    vocab_payload = json.load(f)

species_to_idx = vocab_payload["species_to_idx"]
idx_to_species = {int(k): v for k, v in vocab_payload["idx_to_species"].items()}
num_classes = len(species_to_idx)

# Preload ResNet-18 Backbone
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading Pretrained ResNet-18 on: {device}...")

base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
feature_extractor = nn.Sequential(*list(base_model.children())[:-1]).to(device) # Up to avgpool
feature_extractor.eval()

def extract_features_for_manifest(df: pd.DataFrame, batch_size=64):
    print(f"Extracting 512-dim ResNet-18 embeddings for {len(df)} samples...")
    t0 = time.time()
    all_features = []
    all_labels = []

    # Batch process
    for i in range(0, len(df), batch_size):
        batch_rows = df.iloc[i : i + batch_size]
        tensors = []
        for _, row in batch_rows.iterrows():
            npy_path = CACHE_DIR / (row['filename'].replace(".ogg", ".npy"))
            mel = np.load(npy_path)
            mel_norm = (mel - mel.mean()) / (mel.std() + 1e-6)
            tensor_3c = torch.from_numpy(mel_norm).float().repeat(3, 1, 1) # (3, 128, 313)
            tensors.append(tensor_3c)
        
        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = feature_extractor(batch_tensor).squeeze(-1).squeeze(-1) # (B, 512)
        all_features.append(feats.cpu().numpy())
        all_labels.extend(batch_rows['species_idx'].tolist())

    X = np.concatenate(all_features, axis=0)
    y = np.array(all_labels, dtype=np.int64)
    print(f"Extraction completed in {time.time()-t0:.2f}s. Shape: {X.shape}")
    return X, y

X_train, y_train = extract_features_for_manifest(train_df)
X_val, y_val = extract_features_for_manifest(val_df)
X_test, y_test = extract_features_for_manifest(test_df)

# Deep Bioacoustic Classification Head
class BioacousticDeepHead(nn.Module):
    def __init__(self, in_features=512, hidden_dim=256, num_classes=15, dropout_rate=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(), # Swish activation for smooth gradient dynamics
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(p=0.30),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

class FeatureDataset(Dataset):
    def __init__(self, X, y, is_train=False):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.is_train = is_train
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        # Train-time feature jittering / dropout
        if self.is_train and (np.random.rand() > 0.5):
            noise = torch.randn_like(x) * 0.05
            x = x + noise
        return x, y

train_loader = DataLoader(FeatureDataset(X_train, y_train, is_train=True), batch_size=32, shuffle=True)
val_loader = DataLoader(FeatureDataset(X_val, y_val, is_train=False), batch_size=32, shuffle=False)
test_loader = DataLoader(FeatureDataset(X_test, y_test, is_train=False), batch_size=32, shuffle=False)

head = BioacousticDeepHead(in_features=512, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-5)

print("\n" + "=" * 80)
print("TRAINING OPTIMIZED PRETRAINED RESNET-18 DEEP CLASSIFICATION HEAD")
print("=" * 80)
print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Train Acc':<10} | {'Val Acc':<10} | {'Val Macro F1':<12}")
print("=" * 80)

best_val_f1 = -1.0
best_epoch = 0
best_model_weights = None
history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

for epoch in range(1, 36):
    head.train()
    running_loss = 0.0
    t_preds, t_targets = [], []
    for x_b, y_b in train_loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = head(x_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * len(y_b)
        t_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
        t_targets.extend(y_b.cpu().numpy())
        
    scheduler.step()
    train_loss = running_loss / len(X_train)
    train_acc = accuracy_score(t_targets, t_preds)
    
    # Validation
    head.eval()
    val_loss = 0.0
    v_preds, v_targets = [], []
    with torch.no_grad():
        for x_b, y_b in val_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            logits = head(x_b)
            loss = criterion(logits, y_b)
            val_loss += loss.item() * len(y_b)
            v_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            v_targets.extend(y_b.cpu().numpy())
            
    val_loss /= len(X_val)
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
        best_model_weights = head.state_dict().copy()
        
    if epoch % 5 == 0 or is_best or epoch == 35:
        print(f"{epoch:>2}/35  | {train_loss:>10.4f} | {val_loss:>10.4f} | {train_acc*100:>9.2f}% | {val_acc*100:>9.2f}% | {val_f1:>12.4f}{marker}")

# Save Best Model Pipeline
full_model = nn.Sequential(
    feature_extractor,
    head
)
head.load_state_dict(best_model_weights)

torch.save({
    "epoch": best_epoch,
    "head_state_dict": head.state_dict(),
    "best_val_f1": best_val_f1,
    "architecture": "PretrainedResNet18_DeepHead"
}, OUTPUT_DIR / "best_optimized_resnet18.pt")

# =========================================================================
# HOLDOUT TEST SET EVALUATION
# =========================================================================
print("\n" + "=" * 80)
print(f"EVALUATING OPTIMIZED MODEL ON HOLDOUT TEST SET (N={len(X_test)})")
print("=" * 80)

head.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for x_b, y_b in test_loader:
        x_b = x_b.to(device)
        logits = head(x_b)
        test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        test_targets.extend(y_b.numpy())

test_acc = accuracy_score(test_targets, test_preds)
test_macro_f1 = f1_score(test_targets, test_preds, average='macro', zero_division=0)
test_weighted_f1 = f1_score(test_targets, test_preds, average='weighted', zero_division=0)

print(f"Optimized Test Accuracy   : {test_acc*100:.2f}%")
print(f"Optimized Test Macro F1   : {test_macro_f1:.4f} ({test_macro_f1*100:.2f}%)")
print(f"Optimized Test Weighted F1: {test_weighted_f1:.4f} ({test_weighted_f1*100:.2f}%)")
print("=" * 80)

target_names = [f"{idx_to_species[i]} ({vocab_payload['species_metadata'][idx_to_species[i]]['common_name']})" for i in range(num_classes)]
cls_report = classification_report(test_targets, test_preds, target_names=target_names, digits=4, zero_division=0)
print(cls_report)

# Save Evaluation Report JSON
eval_report = {
    "model_name": "Optimized Pretrained ResNet-18 (Deep Bioacoustic Head)",
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
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=short_names, yticklabels=short_names, square=True)
plt.title(f"Optimized ResNet-18 Test Confusion Matrix (Overall Accuracy: {test_acc*100:.2f}%)", fontsize=14, fontweight='bold', pad=15)
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
print("All artifacts successfully saved!")
