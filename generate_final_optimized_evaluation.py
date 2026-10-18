"""
Generate complete final evaluation report and confusion matrix for the
Top-3 Multi-Window Bioacoustic Attention Network on the holdout test set (N=338).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torchvision.models as models
import librosa
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

VOCAB_PATH = Path("data_artifacts/vocab/species_mapping.json")
TEST_MANIFEST = Path("data_artifacts/manifests/test_manifest.csv")
MODEL_PATH = Path("model_artifacts/optimized_mobilenet/best_mobilenet_model.pt")
AUDIO_BASE = Path("sample_dataset/audio")
OUTPUT_DIR = Path("model_artifacts/optimized_mobilenet")

with open(VOCAB_PATH) as f:
    vocab = json.load(f)

num_classes = vocab["num_classes"]
idx_to_species = {int(k): v for k, v in vocab["idx_to_species"].items()}
species_metadata = vocab["species_metadata"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
m = models.mobilenet_v3_small(weights=None)
in_f = m.classifier[0].in_features
m.classifier = torch.nn.Sequential(
    torch.nn.Linear(in_f, 256, bias=False),
    torch.nn.BatchNorm1d(256),
    torch.nn.Hardswish(),
    torch.nn.Dropout(p=0.35),
    torch.nn.Linear(256, num_classes)
)
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
m.load_state_dict(ckpt["model_state_dict"])
m.to(device).eval()

test_df = pd.read_csv(TEST_MANIFEST)

def get_top_k_windows(y, sr=32000, duration=5.0, k=3):
    w = int(sr * duration)
    if len(y) <= w:
        reps = int(np.ceil(w / len(y)))
        return [np.tile(y, reps)[:w]]
    hop = int(sr * 1.0)
    windows = []
    energies = []
    for s in range(0, len(y) - w + 1, hop):
        seg = y[s : s + w]
        rms = np.sqrt(np.mean(seg**2))
        windows.append(seg)
        energies.append(rms)
    top_indices = np.argsort(energies)[::-1][:k]
    return [windows[i] for i in top_indices]

def extract_mel(clip, sr=32000):
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
    return torch.from_numpy(mel_norm).float().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

all_preds = []
all_targets = []
all_probs = []

print("Running final multi-window inference across all 338 test recordings...")
for _, row in test_df.iterrows():
    p = AUDIO_BASE / row["filename"]
    y, sr = librosa.load(str(p), sr=32000, mono=True)
    clips = get_top_k_windows(y, sr=32000, k=3)
    tensors = torch.cat([extract_mel(c) for c in clips], dim=0).to(device)
    with torch.no_grad():
        probs = torch.softmax(m(tensors), dim=1).mean(dim=0).cpu().numpy()
        pred = int(np.argmax(probs))
    all_preds.append(pred)
    all_targets.append(int(row["species_idx"]))
    all_probs.append(probs)

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
all_probs = np.array(all_probs)

# Top-3 Accuracy
top3_hits = 0
for i in range(len(all_targets)):
    cands = np.argsort(all_probs[i])[::-1][:3]
    if all_targets[i] in cands:
        top3_hits += 1
top3_acc = top3_hits / len(all_targets)

acc = accuracy_score(all_targets, all_preds)
macro_p = precision_score(all_targets, all_preds, average='macro', zero_division=0)
macro_r = recall_score(all_targets, all_preds, average='macro', zero_division=0)
macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

print("\n" + "=" * 80)
print("FINAL OPTIMIZED MODEL EVALUATION BENCHMARK (HOLDOUT TEST SET N=338)")
print("=" * 80)
print(f"Top-1 Test Accuracy   : {acc*100:.2f}%")
print(f"Top-3 Test Accuracy   : {top3_acc*100:.2f}%")
print(f"Macro Precision       : {macro_p:.4f} ({macro_p*100:.2f}%)")
print(f"Macro Recall          : {macro_r:.4f} ({macro_r*100:.2f}%)")
print(f"Macro F1 Score        : {macro_f1:.4f} ({macro_f1*100:.2f}%)")
print(f"Weighted F1 Score     : {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")
print("=" * 80)

target_names = [f"{idx_to_species[i]} ({species_metadata[idx_to_species[i]]['common_name']})" for i in range(num_classes)]
cls_rep = classification_report(all_targets, all_preds, target_names=target_names, digits=4, zero_division=0)
print(cls_rep)

# Save Report JSON
report_dict = {
    "model_name": "Final Optimized Bioacoustic Model (MobileNetV3 + SE-Attention + Multi-Window Inference)",
    "test_accuracy_top1": round(acc, 4),
    "test_accuracy_top3": round(top3_acc, 4),
    "test_macro_precision": round(macro_p, 4),
    "test_macro_recall": round(macro_r, 4),
    "test_macro_f1": round(macro_f1, 4),
    "test_weighted_f1": round(weighted_f1, 4),
    "classification_report": classification_report(all_targets, all_preds, target_names=target_names, output_dict=True, zero_division=0)
}

with open(OUTPUT_DIR / "final_evaluation_report.json", "w") as f:
    json.dump(report_dict, f, indent=4)
print(f"Saved: {OUTPUT_DIR / 'final_evaluation_report.json'}")

# Generate and save confusion matrix
cm = confusion_matrix(all_targets, all_preds)
short_labels = [idx_to_species[i] for i in range(num_classes)]

plt.figure(figsize=(13, 11))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=short_labels,
    yticklabels=[f"{sp} - {species_metadata[sp]['common_name'][:16]}" for sp in short_labels],
    square=True, cbar_kws={'label': 'Test Count'}
)
plt.title(f"Final Optimized Model Holdout Test Confusion Matrix (Accuracy: {acc*100:.2f}%)", fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Predicted Species Code", fontsize=11, fontweight='bold')
plt.ylabel("True Species (Ground Truth)", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "final_confusion_matrix.png", dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'final_confusion_matrix.png'}")
