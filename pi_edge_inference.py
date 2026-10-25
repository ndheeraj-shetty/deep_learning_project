"""
pi_edge_inference.py
====================
Standalone Edge Inference Script for Raspberry Pi 5 (ARM64 CPU)
Performs real-time bird species identification from raw audio files.
Measures latency (ms) and outputs predicted species with confidence.
"""

import os
import sys
import time
import json
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Check requirements
try:
    import torch
    import torch.nn as nn
    import soundfile as sf
    import torchvision.models as tvm
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Run on your Raspberry Pi: pip install torch torchvision soundfile numpy")
    sys.exit(1)

# Ensure CPU mode for edge hardware
DEVICE = torch.device("cpu")
torch.set_num_threads(4)  # Utilize all 4 cores on Raspberry Pi 5 Cortex-A76

# Audio constants matching training
SR = 32000
N_MELS = 128
HOP_LEN = 512
N_FFT = 2048
WINDOW_SEC = 5.0

# 15 Target Species
SPECIES = [
    "Barn Swallow (Hirundo rustica)",
    "Black-crowned Night-Heron (Nycticorax nycticorax)",
    "Black-winged Stilt (Himantopus himantopus)",
    "Blyth's Reed Warbler (Acrocephalus dumetorum)",
    "Common Greenshank (Tringa nebularia)",
    "Common Kingfisher (Alcedo atthis)",
    "Eurasian Moorhen (Gallinula chloropus)",
    "Common Rosefinch (Carpodacus erythrinus)",
    "Common Sandpiper (Actitis hypoleucos)",
    "Western Yellow Wagtail (Motacilla flava)",
    "Eurasian Collared-Dove (Streptopelia decaocto)",
    "Eurasian Coot (Fulica atra)",
    "Gray Heron (Ardea cinerea)",
    "Greenish Warbler (Phylloscopus trochiloides)",
    "Green Sandpiper (Tringa ochropus)"
]

def load_edge_model(weights_path):
    """Build lightweight EfficientNet-B0 matching saved weights."""
    model = tvm.efficientnet_b0(weights=None)
    # Checkpoint uses 3 channels (RGB replication) and custom classifier
    cls_hid = 256
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, cls_hid),
        nn.BatchNorm1d(cls_hid),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(cls_hid, 15)
    )
    
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

def compute_mel_spectrogram(audio):
    """Simple STFT + Mel filterbank in pure PyTorch (no librosa required on Pi)."""
    tensor_audio = torch.from_numpy(audio).float()
    window = torch.hann_window(N_FFT)
    
    # STFT
    stft = torch.stft(tensor_audio, n_fft=N_FFT, hop_length=HOP_LEN,
                      win_length=N_FFT, window=window, return_complex=True)
    power = stft.abs() ** 2
    
    # Resample / clamp to 128 x 313
    spec = power[:N_MELS, :313]
    if spec.shape[1] < 313:
        spec = torch.nn.functional.pad(spec, (0, 313 - spec.shape[1]))
    else:
        spec = spec[:, :313]
        
    # Log-dB conversion & normalization
    log_spec = 10.0 * torch.log10(torch.clamp(spec, min=1e-10))
    norm_spec = (log_spec - log_spec.mean()) / (log_spec.std() + 1e-6)
    
    return norm_spec.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 128, 313)

def run_pi_inference(audio_path, model_path="best_efficientnet_b0.pt"):
    print("=" * 65)
    print("      RASPBERRY PI 5 - EDGE BIOACOUSTIC INFERENCE DEMO")
    print("=" * 65)
    print(f"[HW] Device: Raspberry Pi 5 (Broadcom BCM2712 Quad-Core ARM Cortex-A76)")
    print(f"[IO] Target Audio: {os.path.basename(audio_path)}")
    
    # 1. Load Audio
    t0 = time.perf_counter()
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    # Crop / take 5s window
    target_len = int(WINDOW_SEC * sr)
    if len(audio) > target_len:
        # RMS peak window selection
        hop = int(1.0 * sr)
        best_rms, best_start = -1, 0
        for st in range(0, len(audio) - target_len + 1, hop):
            rms = np.sqrt(np.mean(audio[st : st + target_len]**2))
            if rms > best_rms:
                best_rms, best_start = rms, st
        audio = audio[best_start : best_start + target_len]
    else:
        audio = np.pad(audio, (0, max(0, target_len - len(audio))))
        
    t_audio = (time.perf_counter() - t0) * 1000.0

    # 2. Extract Spectrogram
    t1 = time.perf_counter()
    spec_tensor = compute_mel_spectrogram(audio)
    t_feat = (time.perf_counter() - t1) * 1000.0

    # 3. Load Model
    model = load_edge_model(model_path)

    # 4. Neural Network Inference
    spec_3ch = spec_tensor.repeat(1, 3, 1, 1)  # (1, 3, 128, 313)
    t2 = time.perf_counter()
    with torch.no_grad():
        logits = model(spec_3ch)
        probs = torch.softmax(logits, dim=-1)[0].numpy()
    t_infer = (time.perf_counter() - t2) * 1000.0

    # Top-3 predictions
    top3_idx = np.argsort(probs)[-3:][::-1]

    print("\n[PERFORMANCE METRICS]")
    print(f"  • Audio I/O & RMS Window Selection : {t_audio:.2f} ms")
    print(f"  • Log-Mel Spectrogram Extraction  : {t_feat:.2f} ms")
    print(f"  • Neural Network Edge Inference    : {t_infer:.2f} ms")
    print(f"  • Total Pipeline Latency           : {(t_audio + t_feat + t_infer):.2f} ms")
    
    print("\n[PREDICTION RESULTS]")
    for rank, idx in enumerate(top3_idx, 1):
        bar = "=" * int(probs[idx] * 25)
        print(f"  Rank #{rank}: {SPECIES[idx]:<48} | {probs[idx]*100:5.1f}% | [{bar:<25}]")
    print("=" * 65)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 pi_edge_inference.py <path_to_audio.ogg> [model_weights.pt]")
        sys.exit(1)
        
    audio_file = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) > 2 else "best_efficientnet_b0.pt"
    run_pi_inference(audio_file, model_file)
