# Specification: Hybrid Dual-Signal Turn Detection Model for Voice AI

**Document Version:** 1.0.0  
**Date:** 2026-08-20  
**Status:** Approved  
**Target Environment:** Python 3.10+, PyTorch, LightGBM, Gradio, Librosa, HuggingFace Datasets  

---

## 1. Executive Summary & Problem Framing

Turn detection in voice AI agents determines whether a human speaker has finished their turn (`TURN`) or is merely pausing mid-utterance (`HOLD`). 

### Core Architectural Goal
Achieve ultra-low latency (<5ms CPU) for straightforward turn decisions while leveraging a deep audio encoder (Whisper Tiny) for prosodic and acoustic edge cases, executing within an asymmetric confidence-gated hybrid cascade.

### Evaluation & Constraints
1. **Primary Ground Truth**: `pipecat-ai/smart-turn-data-v3.2` dataset. Evaluated via F1 score, False Positive Rate (FPR), False Negative Rate (FNR), and p50/p95 latency.
2. **Hinglish Evaluation Strategy**: Proxy evaluation utilizing cross-lingual discourse markers ("basically", "right", "so", "okay", "like"), synthetic gTTS filler injections ("acha", "matlab", "toh"), and F0 pitch shifting (+1.5 to +3 semitones for 5-10% of samples). Evaluated and reported explicitly as synthetic proxy metrics.
3. **Latency Target**: Fast path execution budget ≤ 4.2ms CPU total. Max buffer override hard-stop at 8.0 seconds.

---

## 2. System Architecture & Asymmetric Decision Tree

The system uses a 3-stage asymmetric decision tree to process trailing audio windows (default window duration: 1.5 seconds, swept across 0.5s to 2.5s).

```
                           [ Audio Trailing Window ]
                                      │
                         Fast Path Feature Extractor
                       (19 Features via librosa.pyin)
                                      │
                          LightGBM Model → P_fast
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
  P_fast > 0.82                P_fast < 0.20               0.20 ≤ P_fast ≤ 0.82
[STAGE 1: TURN]              [STAGE 1: HOLD]             [ENTER DEAD BAND]
(Skip Slow Path)             (Skip Slow Path)                      │
                                                                   ▼
                                                       Whisper Tiny Encoder (Frozen)
                                                       Pool Last 25% Hidden States
                                                       Linear Head → P_slow
                                                                   │
                                     ┌─────────────────────────────┼─────────────────────────────┐
                                     ▼                             ▼                             ▼
                              P_slow > 0.82                 P_slow < 0.20                 0.20 ≤ P_slow ≤ 0.82
                             [STAGE 2: TURN]               [STAGE 2: HOLD]               [STAGE 3: RESCUE]
                                                                                                 │
                                                                                       P_final = P_fast
                                                                                       P_final ≥ 0.50 ? TURN : HOLD
                                                                                       double_uncertainty_count += 1
```

### Stage Decision Logic
1. **Hard Override**: If cumulative turn buffer duration reaches ≥ 8.0 seconds, force `TURN` immediately.
2. **Stage 1 (Fast Path Alone)**:
   - If `P_fast > 0.82`: Return `TURN` (confidence high, skip slow path).
   - If `P_fast < 0.20`: Return `HOLD` (confidence high, skip slow path).
3. **Stage 2 (Slow Path Override)**:
   - Evaluated when `0.20 ≤ P_fast ≤ 0.82`.
   - Compute `P_slow` via Whisper Tiny Encoder + Linear Head.
   - If `P_slow > 0.82`: Return `TURN`.
   - If `P_slow < 0.20`: Return `HOLD`.
4. **Stage 3 (Double-Uncertainty Fast Path Rescue)**:
   - Evaluated when `0.20 ≤ P_slow ≤ 0.82`.
   - Set `P_final = P_fast`.
   - If `P_final ≥ 0.50`: Return `TURN`. Else: Return `HOLD`.
   - Increment metric logger: `double_uncertainty_count += 1`.

---

## 3. Fast Path Specifications (`features/acoustic.py` & `models/fast_path.py`)

### 3.1 Feature Extraction Pipeline (19 Features)
Computed in a single pass over the normalized floating-point audio array using `librosa` and `numpy`:

1. **F0 Features (6)**:
   - `f0_mean`: Mean of non-zero F0 values extracted via `librosa.pyin(fmin=65, fmax=2093)`.
   - `f0_std`: Standard deviation of non-zero F0 values. (Fallback: drop if extraction > 3.0ms).
   - `f0_slope`: Linear regression slope across non-zero F0 values over time.
   - `f0_final_frame`: F0 value of the final voiced audio frame.
   - `voiced_frame_ratio`: Fraction of total frames containing valid F0 pitch.
   - `f0_final_minus_mean`: `(f0_final_frame - f0_mean)` (Signed intonation direction delta).
2. **Energy Features (5)**:
   - `rms_mean`: Mean root-mean-square energy across frames.
   - `rms_std`: Standard deviation of RMS energy across frames.
   - `rms_slope`: Linear regression slope of RMS energy over time.
   - `rms_final_frame`: RMS energy of the final frame.
   - `rms_tail_ratio`: Ratio of mean RMS energy in the final 25% of frames to the first 75% of frames.
3. **Silence Features (4)**:
   - `trailing_silence_duration`: Duration (in seconds) of continuous trailing silence below energy threshold (threshold = `0.01 * rms_mean`).
   - `silence_ratio`: Fraction of frames with RMS < threshold.
   - `longest_silence_run`: Max consecutive silent frame count.
   - `silence_threshold_crossings`: Number of times energy crosses the silence threshold.
4. **Spectral Features (2)**:
   - `spectral_centroid_mean`: Mean spectral centroid across window frames.
   - `zcr_mean`: Mean zero-crossing rate across window frames.

### 3.2 Fast Path Classifier Configuration
- **Model**: `lightgbm.LGBMClassifier`
- **Parameters**: `n_estimators=1000`, `learning_rate=0.05`, `max_depth=6`, `class_weight='balanced'`, `early_stopping_rounds=50`.
- **Target Execution Budget**: pyin ~3.0ms + numpy ~0.8ms + LGBM ~0.4ms = **~4.2ms Total**.

---

## 4. Slow Path Specifications (`models/slow_path.py`)

### 4.1 Feature Extraction & Pooling
- **Backbone**: `openai/whisper-tiny` (Encoder only, frozen parameters, `in_channels=80`, `d_model=384`).
- **Input Mel-Spectrogram**: Log-mel spectrogram computed from trailing audio window.
- **Window Length Hyperparameter Sweep**: Evaluated across `[0.5s, 1.0s, 1.5s, 2.0s, 2.5s]`. Baseline default: `1.5s`.
- **Trailing-Frame Pooling Logic**:
  Given encoder output tensor $H \in \mathbb{R}^{B \times T \times 384}$:
  $$\text{pooled\_embedding} = \frac{1}{T - \lfloor 0.75 T \rfloor} \sum_{t=\lfloor 0.75 T \rfloor}^{T} H[:, t, :]$$
  *(Mean pool over the final 25% of hidden states. Ablated against full mean-pool in final report).*

### 4.2 Linear Head Architecture
- `nn.Sequential(`
  - `nn.Linear(384, 64)`,
  - `nn.ReLU()`,
  - `nn.Dropout(0.2)`,
  - `nn.Linear(64, 1)`,
  - `nn.Sigmoid()`
- `)`

---

## 5. Hinglish Validator & Synthetic Data Pipeline (`data/augment.py` & `evaluate.py`)

1. **Rule-Based Validator**: Post-fusion heuristic checking for trailing Hinglish discourse markers and filler words. Evaluated qualitatively on a set of 20 manual edge-case recordings.
2. **Synthetic Data Augmentation**:
   - *Filler Injections*: Inject gTTS-generated audio for "acha", "matlab", and "toh" at natural pause boundaries in `HOLD` samples.
   - *Pitch Modulation*: Shift F0 upwards by +1.5 to +3.0 semitones on 5–10% of training samples to model Indian English rising declarative intonation patterns.

---

## 6. Dataset, Training & Logging Pipeline

### 6.1 Dataset Split Protocol (`data/download_dataset.py`)
- Runtime split detection:
  ```python
  try:
      test_ds = load_dataset("pipecat-ai/smart-turn-data-v3.2-test")
      USE_PIPECAT_TEST = True
  except Exception:
      USE_PIPECAT_TEST = False
  ```
- If `USE_PIPECAT_TEST` is True: Use official test split. Train/Val split is 80/20 on `pipecat-ai/smart-turn-data-v3.2-train`.
- If `USE_PIPECAT_TEST` is False: Create stratified 70% Train / 15% Validation / 15% Test splits from `pipecat-ai/smart-turn-data-v3.2-train`.

### 6.2 Loss Function Selection Protocol
Compute overall turn class ratio: $\text{turn\_ratio} = \frac{N_{\text{TURN}}}{N_{\text{TOTAL}}}$.
- If $\text{turn\_ratio} < 0.40$ or $\text{turn\_ratio} > 0.60$: Use `FocalLoss(alpha=0.25, gamma=2.0)`.
- Else: Use standard Binary Cross-Entropy `nn.BCELoss()`.

### 6.3 Audit Logging (`training_log.json`)
Every training run automatically generates `training_log.json` containing:
- `test_split_strategy`: `"scenario_a"` or `"scenario_b"`
- `class_balance_ratio`: float value of $\text{turn\_ratio}$
- `loss_function_used`: `"FocalLoss(gamma=2.0)"` or `"BCELoss"`
- `fast_path_best_iteration`: int
- `fast_path_val_f1`: float
- `slow_path_window_sweep_results`: list of dicts `{window_sec, val_f1, latency_p95_ms}`
- `optuna_triggered`: bool
- `double_uncertainty_count`: int

---

## 7. Gradio Application Interface (`demo.py`)

### Tab 1: Interactive Analysis (Single-Column Mobile Layout)
- **Row 1**: Audio Waveform Display.
- **Row 2**: Dual Probability Plot (`P_fast` thin gray line, `P_slow` thick colored line, dashed red horizontal line at 0.82, dashed blue horizontal line at 0.20).
- **Row 3**: Stage Decision Bar (Green = Stage 1 Fast Path, Yellow = Stage 2 Slow Path, Red = Stage 3 Rescue).
- **Row 4**: Metric Cards (Overall F1, Fast Path Hit Rate %, Slow Path Invocation %, Double Uncertainty Count, Sample p95 Latency).

### Tab 2: Live Mic Demo
- **Controls**: Record/Stop Button, Large TURN/HOLD Indicator, Live Confidence Bar (`P_final`), Stage Text Label, Window Latency (ms).
- **Disclaimer Banner**: `"Live mic latency reflects HuggingFace Spaces CPU — local benchmark is 4.2ms on CPU."`

---

## 8. Directory & File Layout

```
turn-detection/
├── CLAUDE.md
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-08-20-turn-detection-design.md
├── data/
│   ├── download_dataset.py
│   ├── eda.py
│   └── augment.py
├── features/
│   └── acoustic.py
├── models/
│   ├── fast_path.py
│   ├── slow_path.py
│   └── hybrid.py
├── train/
│   ├── train_fast.py
│   ├── train_slow.py
│   └── train_hybrid.py
├── evaluate.py
├── demo.py
├── training_log.json
├── REPORT.md
└── requirements.txt
```
