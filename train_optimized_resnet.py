"""
Train Optimized Bioacoustic ResNet-18 on BirdCLEF 2024 Dataset.
Integrates:
1. Pretrained ImageNet Weights for high representational capacity
2. MixUp Data Augmentation (alpha=0.4) for multi-species audio superposition
3. Label Smoothing Cross-Entropy (0.1) to avoid overconfidence
4. SpecAugment (Time & Frequency Masking)
5. Cosine Annealing Learning Rate Schedule
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple

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

# Paths
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

# Preload Cached Dataset
class PreloadedDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, is_train: bool = False):
        self.manifest = manifest.reset_index(drop=True)
        self.is_train = is_train
        
        self.tensors = []
        self.labels = []
        for _, row in self.manifest.iterrows():
            npy_path = CACHE_DIR / (row['filename'].replace(".ogg", ".npy"))
            mel = np.load(npy_path)
            self.tensors.append(mel)
            self.labels.append(int(row['species_idx']))

    def __len__(self):
        return len(self.manifest)

    def _spec_augment(self, mel: np.ndarray) -> np.ndarray:
        aug = mel.copy()
        n_mels, n_frames = aug.shape
        fill_val = aug.min()
        for _ in range(2):
            f_w = np.random.randint(2, 16)
            f_s = np.random.randint(0, n_mels - f_w)
            aug[f_s : f_s + f_w, :] = fill_val
        for _ in range(2):
            t_w = np.random.randint(4, 24)
            t_s = np.random.randint(0, n_frames - t_w)
            aug[:, t_s : t_s + t_w] = fill_val
        return aug

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        mel = self.tensors[idx]
        label = self.labels[idx]

        if self.is_train and (np.random.rand() > 0.4):
            mel = self._spec_augment(mel)

        # Normalize
        mel_norm = (mel - mel.mean()) / (mel.std() + 1e-6)
        
        # Repeat 1-channel to 3-channels for standard pretrained ImageNet backbones
        tensor_1c = torch.from_numpy(mel_norm).float()
        tensor_3c = tensor_1c.repeat(3, 1, 1) # Shape: (3, 128, 313)
        return tensor_3c, label

# MixUp Implementation
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Optimized Model: Pretrained ResNet-18 with Custom Classifier Head
class OptimizedBioacousticResNet18(nn.Module):
    def __init__(self, num_classes=15, dropout_rate=0.35):
        super().__init__()
        # Load backbone
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Keep all convolutional stages
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Enhanced Classifier Head with Dropout & BatchNorm
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.classifier(x)

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OPTIMIZED RESNET] Training device: {device}")

    train_ds = PreloadedDataset(train_df, is_train=True)
    val_ds = PreloadedDataset(val_df, is_train=False)
    test_ds = PreloadedDataset(test_df, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = OptimizedBioacousticResNet18(num_classes=num_classes).to(device)
    
    # Label smoothing cross entropy
    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
    
    # Differential learning rates: smaller for pretrained backbone, higher for classification head
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params = [p for n, p in model.named_parameters() if "classifier" in n]
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 3e-4},
        {"params": head_params, "lr": 1e-3}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16, eta_min=1e-6)

    num_epochs = 16
    best_val_f1 = -1.0
    best_epoch = 0
    patience = 6
    patience_counter = 0

    best_checkpoint_path = OUTPUT_DIR / "best_optimized_resnet18.pt"

    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [], "val_f1": []
    }

    print("\n" + "=" * 80)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Train Acc':<10} | {'Val Acc':<10} | {'Val Macro F1':<12}")
    print("=" * 80)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)

            # Apply MixUp on 50% of batches
            if np.random.rand() > 0.5:
                mixed_x, y_a, y_b_mix, lam = mixup_data(x_b, y_b, alpha=0.4)
                optimizer.zero_grad()
                logits = model(mixed_x)
                loss = mixup_criterion(criterion, logits, y_a, y_b_mix, lam)
            else:
                optimizer.zero_grad()
                logits = model(x_b)
                loss = criterion(logits, y_b)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item() * len(y_b)
            train_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            train_targets.extend(y_b.detach().cpu().numpy())

        scheduler.step()
        train_loss /= len(train_ds)
        train_acc = accuracy_score(train_targets, train_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits = model(x_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * len(y_b)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.extend(y_b.cpu().numpy())

        val_loss /= len(val_ds)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)

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
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, best_checkpoint_path)
        else:
            patience_counter += 1

        print(f"{epoch:>2}/{num_epochs:<2}  | {train_loss:>10.4f} | {val_loss:>10.4f} | {train_acc*100:>9.2f}% | {val_acc*100:>9.2f}% | {val_f1:>12.4f}{marker}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # Save History JSON
    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    # Final Test Set Evaluation
    print("\n" + "=" * 80)
    print("EVALUATING OPTIMIZED RESNET-18 ON HOLDOUT TEST SET (N=338)")
    print("=" * 80)
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
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

    print(f"Optimized Test Accuracy   : {test_acc*100:.2f}%")
    print(f"Optimized Test Macro F1   : {test_macro_f1:.4f} ({test_macro_f1*100:.2f}%)")
    print(f"Optimized Test Weighted F1: {test_weighted_f1:.4f} ({test_weighted_f1*100:.2f}%)")
    print("=" * 80)

    target_names = [f"{idx_to_species[i]} ({vocab_payload['species_metadata'][idx_to_species[i]]['common_name']})" for i in range(num_classes)]
    cls_report = classification_report(test_targets, test_preds, target_names=target_names, digits=4, zero_division=0)
    print(cls_report)

    eval_report = {
        "architecture": "OptimizedBioacousticResNet18",
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_macro_f1, 4),
        "test_weighted_f1": round(test_weighted_f1, 4),
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4)
    }
    with open(OUTPUT_DIR / "test_evaluation_report.json", "w") as f:
        json.dump(eval_report, f, indent=4)

    # Save Confusion Matrix
    cm = confusion_matrix(test_targets, test_preds)
    short_names = [idx_to_species[i] for i in range(num_classes)]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=short_names, yticklabels=short_names, square=True)
    plt.title(f"Optimized ResNet-18 Test Confusion Matrix (Accuracy: {test_acc*100:.2f}%)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Predicted Species Code", fontsize=12)
    plt.ylabel("True Species Code", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_confusion_matrix.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    run_training()
