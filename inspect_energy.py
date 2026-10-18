import librosa
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv('data_artifacts/manifests/train_manifest.csv')
print('Checking energy distribution across recordings...')
for idx in range(15):
    row = df.iloc[idx]
    p = Path('sample_dataset/audio') / row['filename']
    y, sr = librosa.load(str(p), sr=32000, mono=True)
    dur = len(y) / sr
    window = 5 * 32000
    mid_start = max(0, (len(y) - window) // 2)
    y_mid = y[mid_start : mid_start + window]
    mid_rms = np.sqrt(np.mean(y_mid**2))
    
    if len(y) > window:
        hop = 32000
        rms_steps = [np.sqrt(np.mean(y[s:s+window]**2)) for s in range(0, len(y)-window, hop)]
        best_rms = max(rms_steps)
        best_idx = np.argmax(rms_steps)
    else:
        best_rms = mid_rms
        best_idx = 0
    print(f"File {row['filename']} ({dur:.1f}s): Mid RMS={mid_rms:.4f}, Peak RMS={best_rms:.4f}, Ratio={mid_rms/(best_rms+1e-7):.2f}, Peak at {best_idx}s")
