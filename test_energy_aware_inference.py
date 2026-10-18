"""
Test multi-window energy-aware inference on the holdout test set using the trained MobileNet model.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import librosa
import torch
import torchvision.models as models
from sklearn.metrics import accuracy_score, f1_score, classification_report

VOCAB_PATH = Path("data_artifacts/vocab/species_mapping.json")
TEST_MANIFEST = Path("data_artifacts/manifests/test_manifest.csv")
MODEL_PATH = Path("model_artifacts/optimized_mobilenet/best_mobilenet_model.pt")
AUDIO_BASE = Path("sample_dataset/audio")

with open(VOCAB_PATH) as f:
    vocab = json.load(f)

num_classes = vocab["num_classes"]
idx_to_species = {int(k): v for k, v in vocab["idx_to_species"].items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
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
m.to(device)
m.eval()

test_df = pd.read_csv(TEST_MANIFEST)
print(f"Evaluating {len(test_df)} test recordings using Peak-Energy Window Selection...")

def get_best_energy_clip(y, sr=32000, duration=5.0):
    window = int(sr * duration)
    if len(y) <= window:
        repeats = int(np.ceil(window / len(y)))
        return np.tile(y, repeats)[:window]
    
    # Step by 1 second
    hop = int(sr * 1.0)
    best_start = 0
    max_energy = -1.0
    for start in range(0, len(y) - window + 1, hop):
        segment = y[start : start + window]
        # RMS energy
        rms = np.sqrt(np.mean(segment ** 2))
        if rms > max_energy:
            max_energy = rms
            best_start = start
            
    return y[best_start : best_start + window]

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
    tensor = torch.from_numpy(mel_norm).float().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    return tensor

all_preds = []
all_targets = []

for idx, row in test_df.iterrows():
    audio_path = AUDIO_BASE / row["filename"]
    y, sr = librosa.load(str(audio_path), sr=32000, mono=True)
    clip = get_best_energy_clip(y, sr=32000)
    tensor = extract_mel(clip).to(device)
    with torch.no_grad():
        logits = m(tensor)
        pred = torch.argmax(logits, dim=1).item()
    all_preds.append(pred)
    all_targets.append(int(row["species_idx"]))

acc = accuracy_score(all_targets, all_preds)
macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

print("=" * 80)
print(f"PEAK-ENERGY TEST ACCURACY : {acc*100:.2f}% (was 66.27% on center crop)")
print(f"PEAK-ENERGY TEST MACRO F1 : {macro_f1:.4f} ({macro_f1*100:.2f}%)")
print(f"PEAK-ENERGY TEST WEIGHTED : {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")
print("=" * 80)
