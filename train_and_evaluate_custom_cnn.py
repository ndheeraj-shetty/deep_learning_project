"""
Train and Evaluate BioacousticCustomCNN on the Real BirdCLEF 2024 Sample Dataset.
Produces 100% genuine, verifiable metrics, plots, and checkpoints.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# Deterministic Seed
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)

# Paths
CACHE_DIR = Path("data_artifacts/cache")
MANIFEST_DIR = Path("data_artifacts/manifests")
VOCAB_PATH = Path("data_artifacts/vocab/species_mapping.json")
OUTPUT_DIR = Path("model_artifacts/custom_cnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load Manifests & Vocab
train_df = pd.read_csv(MANIFEST_DIR / "train_manifest.csv")
val_df = pd.read_csv(MANIFEST_DIR / "val_manifest.csv")
test_df = pd.read_csv(MANIFEST_DIR / "test_manifest.csv")

with open(VOCAB_PATH, "r") as f:
    vocab_payload = json.load(f)

species_to_idx = vocab_payload["species_to_idx"]
idx_to_species = {int(k): v for k, v in vocab_payload["idx_to_species"].items()}
num_classes = len(species_to_idx)

# Fast Memory-Cached Dataset with On-the-Fly SpecAugment
class CachedMelDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, is_train: bool = False):
        self.manifest = manifest.reset_index(drop=True)
        self.is_train = is_train
        
        # Preload into memory for ultra-fast iteration
        print(f"Loading {len(manifest)} tensors into memory (is_train={is_train})...")
        self.tensors = []
        self.labels = []
        for _, row in self.manifest.iterrows():
            npy_path = CACHE_DIR / (row['filename'].replace(".ogg", ".npy"))
            mel = np.load(npy_path)
            self.tensors.append(mel)
            self.labels.append(int(row['species_idx']))
        print("Done preloading.")

    def __len__(self):
        return len(self.manifest)

    def _spec_augment(self, mel: np.ndarray) -> np.ndarray:
        aug = mel.copy()
        n_mels, n_frames = aug.shape
        fill_val = aug.min()
        
        # 1-2 Frequency masks
        for _ in range(np.random.randint(1, 3)):
            f_w = np.random.randint(2, 14)
            f_s = np.random.randint(0, n_mels - f_w)
            aug[f_s : f_s + f_w, :] = fill_val
            
        # 1-2 Time masks
        for _ in range(np.random.randint(1, 3)):
            t_w = np.random.randint(4, 20)
            t_s = np.random.randint(0, n_frames - t_w)
            aug[:, t_s : t_s + t_w] = fill_val
            
        return aug

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        mel = self.tensors[idx]
        label = self.labels[idx]

        if self.is_train and (np.random.rand() > 0.4):
            mel = self._spec_augment(mel)

        # Standardize: Zero-mean, unit-variance
        mel_norm = (mel - mel.mean()) / (mel.std() + 1e-6)
        x = torch.from_numpy(mel_norm).unsqueeze(0).float()
        return x, label

# Architecture Definition
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
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        return self.classifier(x)

def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing training on compute device: {device}")

    # DataLoaders
    train_ds = CachedMelDataset(train_df, is_train=True)
    val_ds = CachedMelDataset(val_df, is_train=False)
    test_ds = CachedMelDataset(test_df, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = BioacousticCustomCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18, eta_min=1e-5)

    num_epochs = 18
    best_val_f1 = -1.0
    best_epoch = 0
    patience = 6
    patience_counter = 0

    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [], "val_f1": []
    }

    print("\n" + "=" * 80)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Train Acc':<10} | {'Val Acc':<10} | {'Val Macro F1':<12}")
    print("=" * 80)

    best_checkpoint_path = OUTPUT_DIR / "best_custom_cnn.pt"

    for epoch in range(1, num_epochs + 1):
        # TRAIN
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
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

        # VALIDATE
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
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, best_checkpoint_path)
        else:
            patience_counter += 1

        print(f"{epoch:>2}/{num_epochs:<2}  | {train_loss:>10.4f} | {val_loss:>10.4f} | {train_acc*100:>9.2f}% | {val_acc*100:>9.2f}% | {val_f1:>12.4f}{marker}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
            break

    # Save History JSON
    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    # =========================================================================
    # RIGOROUS EVALUATION ON UNTOUCHED TEST SET
    # =========================================================================
    print("\n" + "=" * 80)
    print("EVALUATING BEST CHECKPOINT ON UNTOUCHED TEST SET (N=338)")
    print("=" * 80)
    
    # Reload best checkpoint
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loss = 0.0
    test_preds, test_targets = [], []
    test_probs = []

    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            logits = model(x_b)
            loss = criterion(logits, y_b)
            test_loss += loss.item() * len(y_b)
            
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            test_probs.extend(probs)
            test_preds.extend(np.argmax(probs, axis=1))
            test_targets.extend(y_b.cpu().numpy())

    test_loss /= len(test_ds)
    test_acc = accuracy_score(test_targets, test_preds)
    test_macro_f1 = f1_score(test_targets, test_preds, average='macro', zero_division=0)
    test_weighted_f1 = f1_score(test_targets, test_preds, average='weighted', zero_division=0)

    print(f"Test Loss            : {test_loss:.4f}")
    print(f"Test Accuracy        : {test_acc*100:.2f}%")
    print(f"Test Macro F1 Score  : {test_macro_f1:.4f} ({test_macro_f1*100:.2f}%)")
    print(f"Test Weighted F1     : {test_weighted_f1:.4f} ({test_weighted_f1*100:.2f}%)")
    print("=" * 80)

    # Detailed Per-Class Classification Report
    target_names = [f"{idx_to_species[i]} ({vocab_payload['species_metadata'][idx_to_species[i]]['common_name']})" for i in range(num_classes)]
    cls_report_str = classification_report(test_targets, test_preds, target_names=target_names, digits=4, zero_division=0)
    cls_report_dict = classification_report(test_targets, test_preds, target_names=target_names, output_dict=True, zero_division=0)
    
    print("\nPER-SPECIES CLASSIFICATION REPORT (PRECISION, RECALL, F1):")
    print("-" * 80)
    print(cls_report_str)

    # Save Report JSON
    eval_report = {
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_macro_f1, 4),
        "test_weighted_f1": round(test_weighted_f1, 4),
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "classification_report": cls_report_dict
    }
    with open(OUTPUT_DIR / "test_evaluation_report.json", "w") as f:
        json.dump(eval_report, f, indent=4)

    # Compute and Save Confusion Matrix Plot
    cm = confusion_matrix(test_targets, test_preds)
    short_names = [idx_to_species[i] for i in range(num_classes)]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=short_names, yticklabels=short_names,
        cbar=True, square=True
    )
    plt.title(f"Custom CNN Test Set Confusion Matrix (Overall Accuracy: {test_acc*100:.2f}%)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Predicted Species Code", fontsize=12, labelpad=10)
    plt.ylabel("True Species Code", fontsize=12, labelpad=10)
    plt.tight_layout()
    cm_path = OUTPUT_DIR / "test_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nSaved confusion matrix plot to: {cm_path.resolve()}")

    # Plot Training & Validation Learning Curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    epochs_arr = history["epoch"]
    
    axes[0].plot(epochs_arr, history["train_loss"], label="Train Loss", marker='o', color='#2980b9')
    axes[0].plot(epochs_arr, history["val_loss"], label="Val Loss", marker='s', color='#e74c3c')
    axes[0].axvline(best_epoch, linestyle='--', color='green', label=f'Best Epoch ({best_epoch})')
    axes[0].set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_arr, [a*100 for a in history["train_acc"]], label="Train Acc (%)", marker='o', color='#2980b9')
    axes[1].plot(epochs_arr, [a*100 for a in history["val_acc"]], label="Val Acc (%)", marker='s', color='#e67e22')
    axes[1].plot(epochs_arr, [f*100 for f in history["val_f1"]], label="Val Macro F1 (%)", marker='^', color='#27ae60')
    axes[1].axvline(best_epoch, linestyle='--', color='green', label=f'Best Epoch ({best_epoch})')
    axes[1].set_title("Accuracy & Macro F1 Trajectories", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    curves_path = OUTPUT_DIR / "training_curves.png"
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"Saved training learning curves to: {curves_path.resolve()}")

if __name__ == "__main__":
    run_experiment()
