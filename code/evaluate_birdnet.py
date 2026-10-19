"""
BirdNET Benchmark Evaluation Script
Evaluates Cornell's BirdNET on the held-out test set (N=338).
Computes open-set and closed-set (15 target species) Top-1, Top-3, Macro F1,
generates confusion matrix and saves evaluation reports.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(encoding='utf-8')

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# LiteRT compatibility layer for birdnetlib
import ai_edge_litert.interpreter as tflite
sys.modules['tflite_runtime.interpreter'] = tflite
sys.modules['tflite_runtime'] = type(sys)('tflite_runtime')
sys.modules['tflite_runtime'].interpreter = tflite

from birdnetlib.analyzer import Analyzer
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_artifacts"
AUDIO_DIR = PROJECT_ROOT / "sample_dataset" / "audio"
MODELS_DIR = PROJECT_ROOT / "model_artifacts" / "birdnet"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load target species vocab
with open(DATA_DIR / "vocab" / "species_mapping.json") as f:
    VOCAB = json.load(f)

NUM_CLASSES = VOCAB["num_classes"]
IDX_TO_SPECIES = {int(k): v for k, v in VOCAB["idx_to_species"].items()}
SPECIES_METADATA = VOCAB["species_metadata"]
CLASS_NAMES = [f"{IDX_TO_SPECIES[i]} ({SPECIES_METADATA[IDX_TO_SPECIES[i]]['common_name']})" for i in range(NUM_CLASSES)]

print("[BIRDNET] Initializing Analyzer...")
analyzer = Analyzer()
birdnet_labels = analyzer.labels
print(f"[BIRDNET] Loaded {len(birdnet_labels)} species classes.")

# Map our 15 classes to BirdNET label indices
class_to_birdnet_idx = {}
for idx, sp_code in IDX_TO_SPECIES.items():
    sci = SPECIES_METADATA[sp_code]["scientific_name"].strip()
    matches = [i for i, lbl in enumerate(birdnet_labels) if sci.lower() in lbl.lower()]
    if not matches:
        raise ValueError(f"Could not map {sp_code} ({sci}) to BirdNET labels")
    class_to_birdnet_idx[idx] = matches[0]

target_birdnet_indices = [class_to_birdnet_idx[i] for i in range(NUM_CLASSES)]
print("[BIRDNET] Successfully mapped all 15 target species to BirdNET label indices.")

# Load Test Manifest
test_df = pd.read_csv(DATA_DIR / "manifests" / "test_manifest.csv")
print(f"[BIRDNET] Evaluating on Holdout Test Set ({len(test_df)} recordings)...")

SAMPLE_RATE = 48000
CHUNK_LEN = 144000  # 3 seconds
HOP_LEN = 72000    # 1.5 seconds

all_targets = []
closed_preds = []
closed_probs = []
open_top1_hits = 0
open_top3_hits = 0

t0 = time.time()
for count, (_, row) in enumerate(test_df.iterrows(), 1):
    fpath = AUDIO_DIR / row["filename"]
    true_idx = int(row["species_idx"])
    true_birdnet_idx = target_birdnet_indices[true_idx]
    
    try:
        y, sr = sf.read(str(fpath), dtype="float32")
        if y.ndim > 1: y = np.mean(y, axis=1)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    except Exception:
        y = np.zeros(CHUNK_LEN, dtype=np.float32)

    # Windowed predictions
    chunk_preds = []
    if len(y) < CHUNK_LEN:
        repeats = int(np.ceil(CHUNK_LEN / max(len(y), 1)))
        chunk = np.tile(y, repeats)[:CHUNK_LEN]
        chunk_preds.append(analyzer.predict(chunk)[0])
    else:
        for start in range(0, len(y) - CHUNK_LEN + 1, HOP_LEN):
            chunk = y[start : start + CHUNK_LEN]
            chunk_preds.append(analyzer.predict(chunk)[0])

    avg_raw_prob = np.mean(chunk_preds, axis=0) # Shape: (6522,)
    
    # 1. Open-Set Metric (among all 6,522 world species)
    open_rank = np.argsort(avg_raw_prob)[::-1]
    if open_rank[0] == true_birdnet_idx:
        open_top1_hits += 1
    if true_birdnet_idx in open_rank[:3]:
        open_top3_hits += 1

    # 2. Closed-Set Metric (restricted to the 15 benchmark species)
    sub_probs = avg_raw_prob[target_birdnet_indices] # Shape: (15,)
    pred_idx = int(np.argmax(sub_probs))
    
    all_targets.append(true_idx)
    closed_preds.append(pred_idx)
    closed_probs.append(sub_probs)
    
    if count % 50 == 0 or count == len(test_df):
        print(f"[{count:03d}/{len(test_df)}] Processed in {time.time()-t0:.1f}s...")

all_targets = np.array(all_targets)
closed_preds = np.array(closed_preds)
closed_probs = np.array(closed_probs)

# Compute metrics
open_acc = open_top1_hits / len(all_targets)
open_top3 = open_top3_hits / len(all_targets)

closed_acc = accuracy_score(all_targets, closed_preds)
closed_top3 = sum(1 for i in range(len(all_targets)) if all_targets[i] in np.argsort(closed_probs[i])[::-1][:3]) / len(all_targets)
closed_macro_f1 = f1_score(all_targets, closed_preds, average="macro", zero_division=0)
closed_weighted_f1 = f1_score(all_targets, closed_preds, average="weighted", zero_division=0)
closed_macro_p = precision_score(all_targets, closed_preds, average="macro", zero_division=0)
closed_macro_r = recall_score(all_targets, closed_preds, average="macro", zero_division=0)

report_data = {
    "model_name": "BirdNET (Cornell / Chemnitz Lab Acoustic SOTA)",
    "benchmark": "BirdCLEF 2024 (15 Target Species)",
    "open_set_top1_accuracy": round(open_acc, 4),
    "open_set_top3_accuracy": round(open_top3, 4),
    "closed_set_top1_accuracy": round(closed_acc, 4),
    "closed_set_top3_accuracy": round(closed_top3, 4),
    "test_accuracy_top1": round(closed_acc, 4),
    "test_accuracy_top3": round(closed_top3, 4),
    "test_macro_precision": round(closed_macro_p, 4),
    "test_macro_recall": round(closed_macro_r, 4),
    "test_macro_f1": round(closed_macro_f1, 4),
    "test_weighted_f1": round(closed_weighted_f1, 4),
    "classification_report": classification_report(all_targets, closed_preds, target_names=CLASS_NAMES, output_dict=True)
}

with open(MODELS_DIR / "final_evaluation_report.json", "w") as f:
    json.dump(report_data, f, indent=4)
with open(MODELS_DIR / "test_evaluation_report.json", "w") as f:
    json.dump(report_data, f, indent=4)

# Confusion Matrix
cm = confusion_matrix(all_targets, closed_preds)
labels = [IDX_TO_SPECIES[i] for i in range(NUM_CLASSES)]
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=labels,
    yticklabels=labels,
    title="Confusion Matrix: BirdNET (Cornell Acoustic Benchmark)",
    ylabel="True Species",
    xlabel="Predicted Species"
)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=8)
plt.tight_layout()
plt.savefig(MODELS_DIR / "final_confusion_matrix.png", dpi=200)
plt.close()

print("\n" + "=" * 80)
print("🏆 BIRDNET EVALUATION COMPLETE!")
print("=" * 80)
print(f"Total Test Samples          : {len(all_targets)}")
print(f"Open-Set Top-1 Accuracy     : {open_acc*100:.2f}% (Out of 6,522 world species)")
print(f"Open-Set Top-3 Accuracy     : {open_top3*100:.2f}%")
print(f"Closed-Set Top-1 Accuracy   : {closed_acc*100:.2f}% (Target 15 species)")
print(f"Closed-Set Top-3 Accuracy   : {closed_top3*100:.2f}%")
print(f"Macro Precision             : {closed_macro_p:.4f} ({closed_macro_p*100:.2f}%)")
print(f"Macro Recall                : {closed_macro_r:.4f} ({closed_macro_r*100:.2f}%)")
print(f"Macro F1 Score              : {closed_macro_f1:.4f} ({closed_macro_f1*100:.2f}%)")
print(f"Weighted F1 Score           : {closed_weighted_f1:.4f} ({closed_weighted_f1*100:.2f}%)")
print("=" * 80 + "\n")
