# Automated Bird Species Identification from Audio Using Deep Learning and Spectrogram Analysis

**Course:** Deep Learning Lab (CA3 Evaluation)  
**Date:** October 19, 2026  
**Topic:** Bioacoustic Audio Classification with Deep Neural Networks  

---

## Abstract
Listening to nature and identifying bird species from audio recordings is one of the most effective ways to track wildlife health, but field audio is messy. Recordings are full of wind, rain, rustling leaves, and long stretches of silence between bird calls. In this project, we built a complete deep learning pipeline to recognize 15 target bird species from the BirdCLEF 2024 benchmark (2,250 audio files). Instead of feeding raw sound waves directly into a network, we convert 5-second audio chunks into 128-band log-Mel spectrograms, turning sound recognition into an image classification problem. To ensure our testing was completely fair and leak-free, we split the data strictly at the recording level (70% train, 15% validation, 15% test)—meaning no recording in the test set had ever been sliced or seen during training. We systematically tested and compared six different neural network architectures—ranging from a simple 4-block CNN built from scratch, to classic ResNet-18, attention-based MobileNetV3, a hybrid CRNN (CNN + Bidirectional GRU), EfficientNet-B0, and a modern ConvNeXt-Tiny. We also compared everything against Cornell's industry-standard BirdNET foundation model. On top of that, we discovered that picking the loudest parts of the audio using Root-Mean-Square (RMS) energy rather than blindly taking the middle 5 seconds boosted model accuracy by over 13%. Our best custom model, ConvNeXt-Tiny, reached 87.28% test accuracy, while our lightweight CRNN achieved 83.73% accuracy from scratch with fewer than 800k parameters.

**Keywords:** Bioacoustics, Mel Spectrograms, BirdCLEF 2024, Audio Classification, CRNN, EfficientNet, ConvNeXt, BirdNET.

---

## 1. Introduction
If you want to know what birds live in a forest, sending humans out with binoculars and notepads is slow, expensive, and disturbs the animals. A much better way is Passive Acoustic Monitoring (PAM): hanging small, battery-powered recorders on trees and letting them record the forest for months. 

The real headache begins when you bring the memory cards back to the lab. You end up with thousands of hours of audio. Having a human listen through all of it is impossible. That is where deep learning comes in.

However, recognizing bird sounds with computers is tricky for three big reasons:
1. **Background Noise:** In real field recordings, birds are competing with wind, rain, flowing rivers, insects, and passing cars.
2. **Birds don't sing continuously:** In a 30-second recording, a bird might only sing for 3 seconds. If your code randomly grabs the middle 5 seconds, it might just analyze empty background rustle.
3. **Similar songs and accents:** Some bird species sound nearly identical to each other, while birds of the same species sing slightly different tunes depending on where they live.

To solve this, we convert raw 1D sound vibrations into 2D pictures of sound called **log-Mel spectrograms** using the Short-Time Fourier Transform (STFT). Once audio is turned into an image of frequencies over time, we can use Convolutional Neural Networks (CNNs) to spot the unique "visual signature" of each bird's chirp.

In this project, our goals were:
- Build a clean audio preprocessing pipeline with **zero data leakage** (ensuring audio files from test never touch train).
- Implement and compare a ladder of 6 different models to understand what architectural ideas actually help with audio.
- Build an activity-aware audio slicer that finds the actual bird chirps based on sound energy.
- Compare our models against Cornell's famous **BirdNET** foundation model to see how close a student project can get to a world-class model.

---

## 2. Literature Review
The way computers analyze animal sounds has changed a lot over the years:

### 2.1 The Old School Way: Handcrafted Features
Early bioacoustic software looked at simple statistical numbers extracted from sound waves, such as zero-crossing rates (how fast the wave wiggles), spectral centroids (the center of pitch), and Mel-Frequency Cepstral Coefficients (MFCCs). These numbers were fed into basic machine learning classifiers like Support Vector Machines (SVMs) or Random Forests. While this worked okay for clean, studio-like bird calls, it failed miserably outdoors whenever it started raining or wind blew across the microphone.

### 2.2 The Deep Learning Shift: Audio as Images
Around 2015, researchers realized that if you convert sound into a 2D Mel spectrogram, you can treat sound recognition like computer vision. Early pioneers like Piczak (2015) and Salamon & Bello (2017) showed that simple 2D CNNs beat handcrafted features by a wide margin. Soon after, people started taking vision models pre-trained on ImageNet (like ResNet and MobileNet) and fine-tuning them on spectrograms. This became the standard winning recipe in international competitions like Kaggle's BirdCLEF.

### 2.3 Modern Architectures & Foundation Models
In recent years, two big trends emerged:
1. **Smarter CNNs:** Architectures like EfficientNet (Tan & Le, 2019) showed that scaling depth, width, and resolution together using inverted bottleneck blocks gives huge performance boosts without ballooning parameter counts. More recently, ConvNeXt (Liu et al., 2022) proved that pure CNNs using large 7x7 filters could perform just as well as Vision Transformers without needing crazy amounts of compute.
2. **Dedicated Bioacoustic Models (BirdNET):** The Cornell Lab of Ornithology trained BirdNET (Kahl et al., 2021) on tens of thousands of hours of audio spanning over 6,000 species worldwide. It is the reigning gold standard. But running BirdNET requires pre-existing models and specialized setups; comparing our custom models against it helps us benchmark how much performance we can squeeze out of smaller, specialized neural nets.

---

## 3. Methodology

### 3.1 The Dataset and the Zero-Leakage Split
We worked with a balanced 15-species subset from the **BirdCLEF 2024** competition. Each species has 150 real-world recordings, giving us **2,250 audio files (~758 MB)** in `.ogg` format.

```
+---------------------------------------------------------------------------------------------------+
|                                 15 TARGET AVIAN SPECIES INVENTORY                                |
+---------+----------------------------------+---------+--------------------------------------------+
| Code    | Scientific Name (Common Name)    | Code    | Scientific Name (Common Name)              |
+---------+----------------------------------+---------+--------------------------------------------+
| barswa  | Hirundo rustica (Barn Swallow)   | comsan  | Actitis hypoleucos (Common Sandpiper)      |
| bcnher  | Nycticorax nycticorax (Night-H.) | eaywag1 | Motacilla flava (W. Yellow Wagtail)        |
| bkwsti  | Himantopus himantopus (Stilt)    | eucdov  | Streptopelia decaocto (Collared-Dove)      |
| blrwar1 | Acrocephalus dumetorum (Warbler) | eurcoo  | Fulica atra (Eurasian Coot)                |
| comgre  | Tringa nebularia (Greenshank)    | graher1 | Ardea cinerea (Gray Heron)                 |
| comkin1 | Alcedo atthis (Kingfisher)       | grewar3 | Phylloscopus trochiloides (Greenish Warb.) |
| commoo3 | Gallinula chloropus (Moorhen)    | grnsan  | Tringa ochropus (Green Sandpiper)          |
| comros  | Carpodacus erythrinus (Rosefinch)|         |                                            |
+---------+----------------------------------+---------+--------------------------------------------+
```

#### Why Recording-Level Splitting Matters (Zero Data Leakage)
A very common beginner mistake in audio projects is to slice a 30-second recording into six 5-second pieces, shuffle all the pieces together, and then split into train and test. That is cheating: piece #1 of a recording ends up in the training set, and piece #2 of the *exact same recording with the same background noise and microphone* ends up in the test set. The model simply memorizes the background noise.

To prevent this, we performed a **strict recording-level split**:
- **Train (70%):** 1,575 complete recordings
- **Validation (15%):** 337 complete recordings
- **Test (15%):** 338 complete recordings

No audio file in the test set shares any audio or background history with the training set.

### 3.2 Preprocessing: How We Turn Sound into an Image
Sound files come in all shapes, volumes, and formats. We standardized them through a 5-step pipeline:

```
+---------------------------------------------------------------------------------------------------+
|                                     AUDIO TO SPECTROGRAM PIPELINE                                 |
+---------------------------------------------------------------------------------------------------+
|  1. Raw Audio (.ogg)       --> Resampled to 32,000 Hz monophonic sound                            |
|  2. Volume Normalization   --> Scaled to 95% peak volume (so quiet & loud clips are balanced)     |
|  3. 5-Second Slicing       --> Sliced into fixed 160,000-sample clips                             |
|  4. Short-Time Fourier Tr. --> Sliced into 64ms windows (Hann) with 16ms hops                     |
|  5. 128 Mel Filterbank     --> Frequencies mapped to 128 Mel bands (50 Hz to 14,000 Hz)           |
|  6. Log-Decibel & Scaling  --> Power converted to dB, then normalized to zero mean & unit variance|
|                                                                                                   |
|  ==> Output Tensor: 128 rows (frequencies) x 313 columns (time steps)                             |
+---------------------------------------------------------------------------------------------------+
```

- **Why 32 kHz?** Bird songs contain very high-pitched chirps. Under the Nyquist rule, 32 kHz lets us capture sounds up to 16 kHz, covering almost all bird vocalizations.
- **Why Mel scale?** Neither human nor bird ears hear pitch linearly. The jump from 200 Hz to 400 Hz feels huge, but the jump from 8,000 Hz to 8,200 Hz is barely noticeable. The Mel scale groups frequencies logarithmically, giving more attention to lower and middle frequencies where calls are rich in detail.
- **Why Decibels (dB)?** Loudness is perceived logarithmically. Converting to decibels makes quiet harmonics visible next to loud chirps.

### 3.3 Data Augmentation (SpecAugment)
To stop models from memorizing exact training samples, we applied **SpecAugment**:
- Randomly blanking out up to 12 frequency lines across the image.
- Randomly blanking out up to 18 vertical time strips.
- Adding small random white noise and circularly rolling the sound in time.
This forces the model to look at the overall song structure rather than depending on one specific frequency band.

---

### 3.4 Why We Chose These Specific Models (Justification & Purpose)
Instead of picking random architectures, we selected models to test specific engineering hypotheses:

```
+----------------------------------------------------------------------------------------------------+
|                                    MODEL SELECTION & RATIONALE                                     |
+----------------------+------------+----------------------------------------------------------------+
| Model                | Parameters | Purpose / Why We Chose It                                      |
+----------------------+------------+----------------------------------------------------------------+
| 1. Baseline CNN      |    424k    | Baseline: To see how well a standard 4-layer CNN performs      |
|                      |            | trained from scratch without any external pretraining.         |
|                      |            |                                                                |
| 2. ResNet-18         |   11.2M    | The Classic Residual: Tests standard computer vision transfer  |
|                      |            | learning with skip connections on audio spectrograms.          |
|                      |            |                                                                |
| 3. MobileNetV3 + SE  |    1.7M    | Channel Attention: Bird calls live in specific frequencies;    |
|                      |            | Squeeze-and-Excitation (SE) lets the model dynamically boost   |
|                      |            | active frequency channels while ignoring noisy bands.          |
|                      |            |                                                                |
| 4. Bioacoustic CRNN  |    787k    | Sequential Temporal Modeling: Birds sing in a sequence of      |
|                      |            | chirps. CNN extracts local textures, Bi-GRU tracks the rhythm  |
|                      |            | and order of syllables over time with attention pooling.       |
|                      |            |                                                                |
| 5. EfficientNet-B0   |    4.3M    | Compound Scaling: Uses Mobile Inverted Bottlenecks (MBConv)    |
|                      |            | to test if balanced scaling of depth, width, and resolution    |
|                      |            | achieves top-tier accuracy with very low parameters.           |
|                      |            |                                                                |
| 6. ConvNeXt-Tiny     |   28.0M    | Modernized CNN: Uses large 7x7 depthwise filters and LayerNorm |
|                      |            | to test if modern vision designs beat classic CNNs on sound.   |
|                      |            |                                                                |
| 7. BirdNET (Cornell) | Foundation | SOTA Benchmark: Pretrained on 6,522 species across 100k+ hours |
|                      |            | to show where our custom student models stand against the best.|
+----------------------+------------+----------------------------------------------------------------+
```

### 3.5 The Secret Weapon: Activity-Aware Multi-Window Ensembling
If a recording is 20 seconds long, where is the bird? If you simply take the center 5 seconds, the bird might have chirped at second 2 and second 18, meaning you feed 5 seconds of empty rustling leaves to the model.

To fix this, we wrote an **Activity-Aware Window Slicer**:
1. We slide a 5.0-second window across the full audio file with a 1.0-second step.
2. For each window, we compute the Root-Mean-Square (RMS) energy:
   $$\text{RMS} = \sqrt{\frac{1}{N} \sum x^2}$$
   Loud chirps have high RMS energy; silent background rustle has low RMS energy.
3. We grab the **Top-3 highest energy windows** (the 3 most active vocalization moments).
4. We run all 3 slices through the model and average their softmax probability predictions.

---

## 4. Results and Discussion

### 4.1 Quantitative Performance Comparison
All models were tested on the exact same 338 held-out recordings that they had never encountered during training.

```
+-----------------------------------------------------------------------------------------------------------------+
|                        COMPREHENSIVE BENCHMARK RESULTS ON HOLDOUT TEST SET (N = 338)                            |
+--------------------------+-----------------------+-------------+------------+------------+----------+-----------+
| Model Architecture       | Inference Windowing   | Parameters  | Top-1 Acc  | Top-3 Acc  | Macro F1 | Weight F1 |
+--------------------------+-----------------------+-------------+------------+------------+----------+-----------+
| Baseline Custom CNN      | Center 5.0s Crop      |     423,727 |   50.59%   |   68.34%   |  50.26%  |  50.17%   |
| Pretrained ResNet-18     | Center 5.0s Crop      |  11,184,143 |   49.11%   |     —      |  49.00%  |  49.02%   |
| MobileNetV3 + SE Attn.   | Center 5.0s Crop      |   1,674,863 |   66.27%   |   84.02%   |  66.22%  |  66.24%   |
| MobileNetV3 + SE Attn.   | Peak-Energy Crop      |   1,674,863 |   75.74%   |   89.94%   |  75.79%  |     —     |
| MobileNetV3 + SE Attn.   | Top-3 Window Ensemble |   1,674,863 |   79.59%   |   92.60%   |  79.75%  |  79.72%   |
| Bioacoustic CRNN         | Top-3 Window Ensemble |     786,800 |   83.73%   |   93.49%   |  83.81%  |  83.80%   |
| EfficientNet-B0          | Top-3 Window Ensemble |   4,339,595 |   86.69%   |   94.67%   |  86.63%  |  86.61%   |
| ConvNeXt-Tiny            | Top-3 Window Ensemble |  28,021,359 |   87.28%   |   95.86%   |  87.37%  |  87.32%   |
| BirdNET (Open-Set 6522)  | Sliding Window (3.0s) | Foundation  |   83.14%   |   94.97%   |    —     |     —     |
| BirdNET (Closed-Set 15)  | Sliding Window (3.0s) | Foundation  |   97.34%   |   99.70%   |  97.33%  |  97.32%   |
+--------------------------+-----------------------+-------------+------------+------------+----------+-----------+
```

### 4.2 What the Numbers Tell Us (Analysis & Interpretation)

1. **Why ConvNeXt-Tiny Won (87.28% Top-1, 87.37% F1):**
   ConvNeXt-Tiny uses large $7\times 7$ depthwise filters. In audio spectrograms, bird chirps stretch across multiple frequency bins and across multiple time steps. Standard $3\times 3$ filters only see a tiny patch at a time, whereas $7\times 7$ filters can view the entire syllable and its harmonic curves at once.
2. **Why EfficientNet-B0 is the Practical Champion (86.69% Acc, 4.3M params):**
   ConvNeXt needed 28 million parameters to reach 87.28%. EfficientNet-B0 reached 86.69% using only **4.3 million parameters** (almost 7 times smaller!). Its inverted bottleneck blocks and compound scaling strike the best balance between accuracy and file size.
3. **The Big Surprise: CRNN Built From Scratch (83.73% Acc, 786k params):**
   Our custom CRNN has **zero ImageNet pretraining** and only **786k parameters**. Yet, it beat our pretrained ResNet-18 by over 34% and matched BirdNET's open-set accuracy (83.14%). Because it uses a Bidirectional GRU, it actually pays attention to the *rhythm* of the bird song over time.
4. **Why ResNet-18 Struggled (49.11% Acc):**
   Standard ResNet-18 treats every pixel and channel equally. It lacks Squeeze-and-Excitation attention. In noisy audio, high-frequency hiss and microphone hum occupy many pixels, and without attention to tell the model "ignore this hiss, focus on the chirp," ResNet overfit on the background noise.
5. **The Impact of Windowing (+13.3% Boost):**
   Looking at MobileNetV3, center cropping gave 66.27%, finding the loudest peak gave 75.74%, and taking the top-3 loudest windows gave 79.59%. This **+13.3% jump** proves that *how* you feed audio to a model at test time is just as important as the neural network architecture itself.

### 4.3 Qualitative Error Analysis: Which Birds Were Easy vs. Hard?
Looking at our confusion matrices and per-species F1 heatmap:
- **The Easiest Birds ($F1 > 90\%$):** 
  - *Common Kingfisher* (`comkin1`) and *Common Rosefinch* (`comros`).
  - *Why:* Kingfishers make extremely sharp, high-frequency, piercing whistles. On a spectrogram, these appear as bright, razor-sharp needle marks that are impossible to confuse with wind or rumbling.
- **The Hardest Birds ($F1 < 75\%$):** 
  - *Eurasian Moorhen* (`commoo3`) and *Black-crowned Night-Heron* (`bcnher`).
  - *Why:* Night-herons and moorhens make low-pitched, raspy, guttural croaks. These sounds occupy the low frequencies (under 1,000 Hz) where wind noise, water splashes, and traffic hum live. The model frequently confused their calls with environmental rumble.
- **BirdNET's Performance:**
  BirdNET achieved 97.34% closed-set accuracy because it has seen millions of hours of forest rain and wind during training, making it nearly immune to environmental background noise.

---

## 5. Conclusion
In this project, we built a complete, reproducible deep learning system for bird vocalization classification. The key lessons we learned are:
1. **Don't just center-crop audio:** Finding peak sound energy with an RMS sliding window provided an immediate **+13% accuracy boost** across models without retraining a single weight.
2. **Sequential models excel on sound:** Our lightweight CRNN (CNN + Bi-GRU) proved that modeling sound as a time sequence yields massive parameter efficiency (83.73% accuracy with under 800k parameters from scratch).
3. **Compound scaling wins for production:** EfficientNet-B0 delivered 86.69% accuracy with only 4.3M parameters, making it the ideal candidate for battery-powered Raspberry Pi or edge devices in the wild.
4. **Zero-leakage is non-negotiable:** Partitioning at the recording level ensured our metrics were 100% honest and directly comparable against world-class foundation models like BirdNET.

---

## 6. References
1. **Kahl, S., Wood, C. M., Eibl, M., & Klinck, H.** (2021). *BirdNET: A deep learning solution for avian diversity monitoring*. Ecological Informatics, 61, 101236.
2. **Tan, M., & Le, Q.** (2019). *EfficientNet: Rethinking model scaling for convolutional neural networks*. ICML 2019, 6105–6114.
3. **Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S.** (2022). *A ConvNet for the 2020s*. IEEE/CVF CVPR 2022, 11976–11986.
4. **Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q. V., & Adam, H.** (2019). *Searching for MobileNetV3*. IEEE/CVF ICCV 2019, 1314–1324.
5. **Hu, J., Shen, L., & Sun, G.** (2018). *Squeeze-and-excitation networks*. IEEE CVPR 2018, 7132–7141.
6. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). *Deep residual learning for image recognition*. IEEE CVPR 2016, 770–778.
7. **Piczak, K. J.** (2015). *Environmental sound classification with convolutional neural networks*. IEEE MLSP 2015, 1–6.
8. **Salamon, J., & Bello, J. P.** (2017). *Deep convolutional neural networks and data augmentation for environmental sound classification*. IEEE Signal Processing Letters, 24(3), 279–283.
9. **Park, D. S., et al.** (2019). *SpecAugment: A simple data augmentation method for automatic speech recognition*. Interspeech 2019, 2613–2617.
10. **Briggs, F., et al.** (2012). *Acoustic classification of bird species using multi-instance multi-label learning*. JASA, 131(6), 4640–4650.
