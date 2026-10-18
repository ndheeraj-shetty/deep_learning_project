# Sample Dataset for Bird Species Identification
**Project:** Deep Learning-Based Automated Bird Species Identification Using Acoustic Signals  
**Benchmark:** BirdCLEF 2024 Curation

## Overview
This directory contains the balanced, curated research dataset comprising **15 target avian species** with **150 audio recordings each** (total **2,250 recordings**, ~758 MB).

This sample dataset is specifically formatted for rapid prototyping, cross-validation, and deep learning model benchmarking (Custom CNN, ResNet, EfficientNet) without needing to download or process the full 15GB+ BirdCLEF 2024 raw dataset.

## Directory Structure
```
sample_dataset/
 ├── audio/                       # 2,250 Ogg Vorbis audio recordings organized by species code
 │    ├── barswa/                 # Barn Swallow (150 audio files)
 │    ├── bcnher/                 # Black-crowned Night-Heron (150 audio files)
 │    ├── bkwsti/                 # Black-winged Stilt (150 audio files)
 │    ├── blrwar1/                # Blyth's Reed Warbler (150 audio files)
 │    ├── comgre/                 # Common Greenshank (150 audio files)
 │    ├── comkin1/                # Common Kingfisher (150 audio files)
 │    ├── commoo3/                # Eurasian Moorhen (150 audio files)
 │    ├── comros/                 # Common Rosefinch (150 audio files)
 │    ├── comsan/                 # Common Sandpiper (150 audio files)
 │    ├── eaywag1/                # Western Yellow Wagtail (150 audio files)
 │    ├── eucdov/                 # Eurasian Collared-Dove (150 audio files)
 │    ├── eurcoo/                 # Eurasian Coot (150 audio files)
 │    ├── graher1/                # Gray Heron (150 audio files)
 │    ├── grewar3/                # Greenish Warbler (150 audio files)
 │    └── grnsan/                 # Green Sandpiper (150 audio files)
 ├── sample_metadata.csv          # Filtered metadata containing 2,250 records with labels and taxonomy
 └── train_metadata.csv           # Direct drop-in alias for automated notebook ingestion
```

## Taxonomic Roster
| # | Species Code (`primary_label`) | Common Name | Scientific Name | Total Recordings |
|---|--------------------------------|-------------|-----------------|------------------|
| 1 | `barswa` | Barn Swallow | *Hirundo rustica* | 150 |
| 2 | `bcnher` | Black-crowned Night-Heron | *Nycticorax nycticorax* | 150 |
| 3 | `bkwsti` | Black-winged Stilt | *Himantopus himantopus* | 150 |
| 4 | `blrwar1` | Blyth's Reed Warbler | *Acrocephalus dumetorum* | 150 |
| 5 | `comgre` | Common Greenshank | *Tringa nebularia* | 150 |
| 6 | `comkin1` | Common Kingfisher | *Alcedo atthis* | 150 |
| 7 | `commoo3` | Eurasian Moorhen | *Gallinula chloropus* | 150 |
| 8 | `comros` | Common Rosefinch | *Carpodacus erythrinus* | 150 |
| 9 | `comsan` | Common Sandpiper | *Actitis hypoleucos* | 150 |
| 10 | `eaywag1` | Western Yellow Wagtail | *Motacilla flava* | 150 |
| 11 | `eucdov` | Eurasian Collared-Dove | *Streptopelia decaocto* | 150 |
| 12 | `eurcoo` | Eurasian Coot | *Fulica atra* | 150 |
| 13 | `graher1` | Gray Heron | *Ardea cinerea* | 150 |
| 14 | `grewar3` | Greenish Warbler | *Phylloscopus trochiloides* | 150 |
| 15 | `grnsan` | Green Sandpiper | *Tringa ochropus* | 150 |

## Acoustic Specifications
- Format: Ogg Vorbis (`.ogg`)
- Default Resampling Rate in Pipeline: 32,000 Hz
- Target Segment Duration: 5.0 seconds (160,000 samples)
- Target Mel Spectrogram Shape: (128 Mel Bins x 313 Time Frames)
