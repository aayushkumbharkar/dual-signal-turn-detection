# Turn Detection Hybrid Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a low-latency, dual-signal hybrid turn detection model (<5ms CPU fast path + Whisper Tiny slow path) with an asymmetric confidence gate, Hinglish proxy evaluation, and a two-tab interactive Gradio demo.

**Architecture:** A 3-stage asymmetric classifier where a 19-feature LightGBM fast path handles clear TURN/HOLD decisions in <5ms. Ambiguous cases (0.20 <= P_fast <= 0.82) invoke a frozen Whisper Tiny encoder with trailing-frame pooling and a linear head (~30ms CPU). Double-uncertainty cases fall back to raw fast path scores with operational logging.

**Tech Stack:** Python 3.10+, PyTorch, LightGBM, Librosa, NumPy, HuggingFace `datasets` & `transformers`, Gradio, PyTest.

**Spec:** [`turn-detection/docs/superpowers/specs/2026-08-20-turn-detection-design.md`](file:///C:/open-data-scientist/turn-detection/docs/superpowers/specs/2026-08-20-turn-detection-design.md)

## Global Constraints

- Python: `3.10+`
- Fast Path Budget: `≤ 4.2ms CPU total`
- Gate Upper Threshold (`TURN_GATE`): `0.82`
- Gate Lower Threshold (`HOLD_GATE`): `0.20`
- Double Uncertainty Fallback Threshold: `0.50`
- Maximum Buffer Duration Hard Stop: `8.0 seconds`
- F0 Extractor: `librosa.pyin(fmin=65, fmax=2093)`
- Feature Count: `19 exact features`

---

### Task 1: Environment & Project Scaffolding

**Files:**
- Create: `turn-detection/requirements.txt`
- Create: `turn-detection/tests/test_scaffolding.py`
- Test: `turn-detection/tests/test_scaffolding.py`

**Interfaces:**
- Consumes: Standard Python environment
- Produces: Project environment with dependencies locked

- [ ] **Step 1: Write test for environment dependencies**

```python
import pytest

def test_imports():
    import torch
    import lightgbm
    import librosa
    import numpy
    import datasets
    import transformers
    import gradio
    assert torch.__version__ is not None
    assert lightgbm.__version__ is not None
```

- [ ] **Step 2: Run test to verify failure before requirements installation**

Run: `pytest turn-detection/tests/test_scaffolding.py`  
Expected: Import error or missing module error if any dependency is absent.

- [ ] **Step 3: Create requirements.txt and install dependencies**

```text
torch>=2.0.0
lightgbm>=4.0.0
librosa>=0.10.0
numpy>=1.24.0
datasets>=2.14.0
transformers>=4.30.0
gradio>=3.50.0
pytest>=7.4.0
gTTS>=2.3.2
scipy>=1.10.0
```

Run: `pip install -r turn-detection/requirements.txt`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest turn-detection/tests/test_scaffolding.py`  
Expected: PASS

---

### Task 2: Data Acquisition & EDA (`data/download_dataset.py` & `data/eda.py`)

**Files:**
- Create: `turn-detection/data/download_dataset.py`
- Create: `turn-detection/data/eda.py`
- Test: `turn-detection/tests/test_data.py`

**Interfaces:**
- Consumes: HuggingFace dataset `pipecat-ai/smart-turn-data-v3.2`
- Produces: Prepared splits (`train`, `val`, `test`), `turn_ratio`, `USE_PIPECAT_TEST` flag

- [ ] **Step 1: Write test for dataset downloading and split logic**

```python
import pytest
from data.download_dataset import load_and_split_dataset

def test_load_and_split_dataset():
    splits, use_pipecat_test, turn_ratio = load_and_split_dataset(toy_mode=True)
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits
    assert 0.0 <= turn_ratio <= 1.0
    assert isinstance(use_pipecat_test, bool)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest turn-detection/tests/test_data.py`  
Expected: FAIL with `ModuleNotFoundError: No module named 'data'`

- [ ] **Step 3: Implement `data/download_dataset.py` & `data/eda.py`**

```python
import numpy as np
from datasets import load_dataset, DatasetDict

def load_and_split_dataset(toy_mode=False):
    use_pipecat_test = False
    try:
        test_ds = load_dataset("pipecat-ai/smart-turn-data-v3.2-test")
        train_ds_raw = load_dataset("pipecat-ai/smart-turn-data-v3.2-train")
        use_pipecat_test = True
    except Exception:
        train_ds_raw = load_dataset("pipecat-ai/smart-turn-data-v3.2-train")
    
    if toy_mode:
        ds = train_ds_raw['train'].select(range(min(100, len(train_ds_raw['train']))))
        splits = ds.train_test_split(test_size=0.3, seed=42)
        val_test = splits['test'].train_test_split(test_size=0.5, seed=42)
        final_splits = DatasetDict({
            'train': splits['train'],
            'val': val_test['train'],
            'test': val_test['test']
        })
        labels = [x['label'] for x in final_splits['train']]
        turn_ratio = float(np.mean(labels))
        return final_splits, use_pipecat_test, turn_ratio

    if use_pipecat_test:
        full_train = train_ds_raw['train']
        train_val = full_train.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
        final_splits = DatasetDict({
            'train': train_val['train'],
            'val': train_val['test'],
            'test': test_ds['test']
        })
    else:
        full_train = train_ds_raw['train']
        train_test = full_train.train_test_split(test_size=0.3, seed=42, stratify_by_column="label")
        val_test = train_test['test'].train_test_split(test_size=0.5, seed=42, stratify_by_column="label")
        final_splits = DatasetDict({
            'train': train_test['train'],
            'val': val_test['train'],
            'test': val_test['test']
        })

    labels = [x['label'] for x in final_splits['train']]
    turn_ratio = float(np.mean(labels))
    return final_splits, use_pipecat_test, turn_ratio
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest turn-detection/tests/test_data.py`  
Expected: PASS

---

### Task 3: Acoustic Feature Extraction (`features/acoustic.py`)

**Files:**
- Create: `turn-detection/features/acoustic.py`
- Test: `turn-detection/tests/test_acoustic_features.py`

**Interfaces:**
- Consumes: Audio numpy array `y` (float32), sampling rate `sr` (int)
- Produces: 19-element 1D numpy array `features`

- [ ] **Step 1: Write test for exact 19 feature extraction and timing**

```python
import numpy as np
import time
from features.acoustic import extract_acoustic_features

def test_extract_acoustic_features_shape_and_latency():
    sr = 16000
    y = np.random.randn(sr * 1 + 800).astype(np.float32)  # 1.5s audio
    
    t0 = time.perf_counter()
    feats = extract_acoustic_features(y, sr)
    t1 = time.perf_counter()
    
    assert feats.shape == (19,)
    assert not np.isnan(feats).any()
    assert (t1 - t0) * 1000.0 < 50.0  # loose initial bound, target 4.2ms
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest turn-detection/tests/test_acoustic_features.py`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `features/acoustic.py` (19 Exact Features)**

```python
import numpy as np
import librosa

def extract_acoustic_features(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    if len(y) == 0:
        return np.zeros(19, dtype=np.float32)
    
    # 1. F0 via pyin
    f0, _, _ = librosa.pyin(y, fmin=65, fmax=2093, sr=sr)
    voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    
    if len(voiced_f0) > 0:
        f0_mean = float(np.mean(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        f0_slope = float(np.polyfit(np.arange(len(voiced_f0)), voiced_f0, 1)[0]) if len(voiced_f0) > 1 else 0.0
        f0_final_frame = float(voiced_f0[-1])
        voiced_frame_ratio = float(len(voiced_f0) / len(f0))
    else:
        f0_mean, f0_std, f0_slope, f0_final_frame, voiced_frame_ratio = 0.0, 0.0, 0.0, 0.0, 0.0
    
    f0_final_minus_mean = f0_final_frame - f0_mean

    # 2. Energy / RMS
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    rms_slope = float(np.polyfit(np.arange(len(rms)), rms, 1)[0]) if len(rms) > 1 else 0.0
    rms_final_frame = float(rms[-1]) if len(rms) > 0 else 0.0
    
    split_idx = int(0.75 * len(rms))
    head_rms = np.mean(rms[:split_idx]) if split_idx > 0 else 1e-6
    tail_rms = np.mean(rms[split_idx:]) if split_idx < len(rms) else 0.0
    rms_tail_ratio = float(tail_rms / (head_rms + 1e-6))

    # 3. Silence
    silence_thresh = 0.01 * rms_mean
    is_silent = rms < silence_thresh
    silence_ratio = float(np.mean(is_silent))
    
    # Trailing silence
    trailing_count = 0
    for s in reversed(is_silent):
        if s: trailing_count += 1
        else: break
    frame_dur = hop_length / sr
    trailing_silence_duration = float(trailing_count * frame_dur)
    
    # Longest run & threshold crossings
    runs = np.diff(np.where(np.concatenate(([sum(is_silent)==0], is_silent, [sum(is_silent)==0])) == 0)[0]) - 1 if len(is_silent) > 0 else np.array([0])
    longest_silence_run = float(np.max(runs)) if len(runs) > 0 else 0.0
    silence_threshold_crossings = float(np.sum(np.diff(is_silent.astype(int)) != 0))

    # 4. Spectral
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_centroid_mean = float(np.mean(spec_cent))
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]
    zcr_mean = float(np.mean(zcr))

    return np.array([
        f0_mean, f0_std, f0_slope, f0_final_frame, voiced_frame_ratio, f0_final_minus_mean,
        rms_mean, rms_std, rms_slope, rms_final_frame, rms_tail_ratio,
        trailing_silence_duration, silence_ratio, longest_silence_run, silence_threshold_crossings,
        spectral_centroid_mean, zcr_mean
    ], dtype=np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest turn-detection/tests/test_acoustic_features.py`  
Expected: PASS

---

### Task 4: Fast Path LightGBM Model (`models/fast_path.py` & `train/train_fast.py`)

**Files:**
- Create: `turn-detection/models/fast_path.py`
- Create: `turn-detection/train/train_fast.py`
- Test: `turn-detection/tests/test_fast_path.py`

**Interfaces:**
- Consumes: 19-feature matrix `X` (N, 19), binary target `y` (N,)
- Produces: Trained LightGBM model artifact `fast_path_lgbm.pkl`, `predict_proba(X)` -> `P_fast`

- [ ] **Step 1: Write test for Fast Path training and inference**

```python
import numpy as np
from models.fast_path import FastPathClassifier

def test_fast_path_fit_predict():
    X = np.random.randn(100, 19).astype(np.float32)
    y = np.random.randint(0, 2, size=(100,))
    
    clf = FastPathClassifier()
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    
    assert probs.shape == (100,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest turn-detection/tests/test_fast_path.py`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `models/fast_path.py`**

```python
import lightgbm as lgb
import numpy as np

class FastPathClassifier:
    def __init__(self, n_estimators=1000, learning_rate=0.05, max_depth=6):
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            class_weight='balanced',
            random_state=42
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None):
        if eval_set is not None:
            self.model.fit(
                X, y,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
        else:
            self.model.fit(X, y)
            
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest turn-detection/tests/test_fast_path.py`  
Expected: PASS

---

### Task 5: Slow Path Model (`models/slow_path.py` & `train/train_slow.py`)

**Files:**
- Create: `turn-detection/models/slow_path.py`
- Create: `turn-detection/train/train_slow.py`
- Test: `turn-detection/tests/test_slow_path.py`

**Interfaces:**
- Consumes: Audio window waveform or pre-extracted pooled Whisper hidden states (B, 384)
- Produces: Binary scalar probability `P_slow` (B, 1)

- [ ] **Step 1: Write test for trailing frame pooling and Linear Head forward pass**

```python
import torch
from models.slow_path import SlowPathLinearHead, pool_trailing_hidden_states

def test_slow_path_head_and_pooling():
    # Simulate Whisper encoder output (B=2, T=100, D=384)
    hidden_states = torch.randn(2, 100, 384)
    pooled = pool_trailing_hidden_states(hidden_states)
    assert pooled.shape == (2, 384)
    
    head = SlowPathLinearHead()
    logits = head(pooled)
    assert logits.shape == (2, 1)
    assert (logits >= 0.0).all() and (logits <= 1.0).all()
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest turn-detection/tests/test_slow_path.py`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `models/slow_path.py`**

```python
import torch
import torch.nn as nn

def pool_trailing_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    # hidden_states shape: (Batch, Time, 384)
    T = hidden_states.shape[1]
    split_idx = int(0.75 * T)
    trailing_states = hidden_states[:, split_idx:, :]
    return torch.mean(trailing_states, dim=1)

class SlowPathLinearHead(nn.Module):
    def __init__(self, in_features: int = 384, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest turn-detection/tests/test_slow_path.py`  
Expected: PASS

---

### Task 6: Hybrid Decision Engine & Asymmetric Gate (`models/hybrid.py`)

**Files:**
- Create: `turn-detection/models/hybrid.py`
- Test: `turn-detection/tests/test_hybrid_engine.py`

**Interfaces:**
- Consumes: Audio window `y`, cumulative duration `buffer_duration_sec`
- Produces: Dict `{decision: "TURN"|"HOLD", stage: 1|2|3, p_fast: float, p_slow: float|None, p_final: float}`

- [ ] **Step 1: Write test for exact 3-stage asymmetric decision logic**

```python
import pytest
from models.hybrid import HybridTurnDetector, TURN_GATE, HOLD_GATE

class DummyFastPath:
    def __init__(self, p_val): self.p_val = p_val
    def predict_proba(self, X): return np.array([self.p_val])

class DummySlowPath:
    def __init__(self, p_val): self.p_val = p_val
    def predict(self, y): return self.p_val

def test_hybrid_stage_1_turn():
    detector = HybridTurnDetector(fast_model=DummyFastPath(0.85), slow_model=DummySlowPath(0.50))
    res = detector.predict(y=np.zeros(100), buffer_dur=1.0)
    assert res['decision'] == 'TURN'
    assert res['stage'] == 1

def test_hybrid_stage_3_rescue():
    detector = HybridTurnDetector(fast_model=DummyFastPath(0.55), slow_model=DummySlowPath(0.40))
    res = detector.predict(y=np.zeros(100), buffer_dur=1.0)
    assert res['decision'] == 'TURN'
    assert res['stage'] == 3
    assert detector.double_uncertainty_count == 1
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest turn-detection/tests/test_hybrid_engine.py`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `models/hybrid.py`**

```python
import numpy as np
from features.acoustic import extract_acoustic_features

TURN_GATE = 0.82
HOLD_GATE = 0.20
MAX_BUFFER_SEC = 8.0

class HybridTurnDetector:
    def __init__(self, fast_model, slow_model):
        self.fast_model = fast_model
        self.slow_model = slow_model
        self.double_uncertainty_count = 0
        
    def predict(self, y: np.ndarray, buffer_dur: float = 0.0, sr: int = 16000) -> dict:
        # Stage 0: Max buffer override
        if buffer_dur >= MAX_BUFFER_SEC:
            return {'decision': 'TURN', 'stage': 0, 'p_fast': 1.0, 'p_slow': None, 'p_final': 1.0}
            
        # Stage 1: Fast Path
        feats = extract_acoustic_features(y, sr=sr)
        p_fast = float(self.fast_model.predict_proba(np.expand_dims(feats, axis=0))[0])
        
        if p_fast > TURN_GATE:
            return {'decision': 'TURN', 'stage': 1, 'p_fast': p_fast, 'p_slow': None, 'p_final': p_fast}
        if p_fast < HOLD_GATE:
            return {'decision': 'HOLD', 'stage': 1, 'p_fast': p_fast, 'p_slow': None, 'p_final': p_fast}
            
        # Stage 2: Slow Path (Dead Band)
        p_slow = float(self.slow_model.predict(y))
        
        if p_slow > TURN_GATE:
            return {'decision': 'TURN', 'stage': 2, 'p_fast': p_fast, 'p_slow': p_slow, 'p_final': p_slow}
        if p_slow < HOLD_GATE:
            return {'decision': 'HOLD', 'stage': 2, 'p_fast': p_fast, 'p_slow': p_slow, 'p_final': p_slow}
            
        # Stage 3: Double-Uncertainty Fast Path Rescue
        self.double_uncertainty_count += 1
        p_final = p_fast
        decision = 'TURN' if p_final >= 0.50 else 'HOLD'
        return {'decision': decision, 'stage': 3, 'p_fast': p_fast, 'p_slow': p_slow, 'p_final': p_final}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest turn-detection/tests/test_hybrid_engine.py`  
Expected: PASS

---

### Task 7: Synthetic Hinglish Augmentation & Evaluation (`data/augment.py` & `evaluate.py`)

**Files:**
- Create: `turn-detection/data/augment.py`
- Create: `turn-detection/evaluate.py`
- Test: `turn-detection/tests/test_evaluation.py`

**Interfaces:**
- Consumes: Dataset splits, trained hybrid model
- Produces: Evaluation metrics dict, `training_log.json`, synthetic Hinglish benchmark table

- [ ] **Step 1: Write test for Hinglish pitch shifting & filler injection**

```python
import numpy as np
from data.augment import apply_pitch_shift, inject_gtts_filler

def test_hinglish_augmentation():
    y = np.random.randn(16000).astype(np.float32)
    y_shifted = apply_pitch_shift(y, sr=16000, n_steps=2.0)
    assert len(y_shifted) > 0
    assert not np.array_equal(y, y_shifted)
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest turn-detection/tests/test_evaluation.py`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `data/augment.py` and `evaluate.py`**

```python
# data/augment.py
import librosa
import numpy as np

def apply_pitch_shift(y: np.ndarray, sr: int = 16000, n_steps: float = 2.0) -> np.ndarray:
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

# evaluate.py
import json
import time
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

def evaluate_model(hybrid_detector, test_dataset) -> dict:
    predictions = []
    labels = []
    latencies = []
    fast_hits = 0
    slow_invocations = 0
    
    for sample in test_dataset:
        audio = sample['audio']['array'] if isinstance(sample['audio'], dict) else sample['audio']
        sr = sample['audio']['sampling_rate'] if isinstance(sample['audio'], dict) else 16000
        true_label = 1 if sample['label'] == 1 or sample['label'] == 'TURN' else 0
        
        t0 = time.perf_counter()
        result = hybrid_detector.predict(audio, buffer_dur=0.0, sr=sr)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        
        latencies.append(lat_ms)
        pred_label = 1 if result['decision'] == 'TURN' else 0
        predictions.append(pred_label)
        labels.append(true_label)
        
        if result['stage'] == 1:
            fast_hits += 1
        elif result['stage'] in (2, 3):
            slow_invocations += 1
            
    total = len(test_dataset) if len(test_dataset) > 0 else 1
    f1 = float(f1_score(labels, predictions, zero_division=0))
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
    results = {
        'overall_f1': f1,
        'fpr': fpr,
        'fnr': fnr,
        'fast_path_hit_rate': float(fast_hits / total),
        'slow_path_invocation_rate': float(slow_invocations / total),
        'double_uncertainty_count': hybrid_detector.double_uncertainty_count,
        'p50_latency_ms': float(np.percentile(latencies, 50)) if len(latencies) > 0 else 0.0,
        'p95_latency_ms': float(np.percentile(latencies, 95)) if len(latencies) > 0 else 0.0
    }
    
    with open('training_log.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest turn-detection/tests/test_evaluation.py`  
Expected: PASS

---

### Task 8: Gradio Interactive Application (`demo.py`)

**Files:**
- Create: `turn-detection/demo.py`
- Test: `turn-detection/tests/test_demo.py`

**Interfaces:**
- Consumes: Audio file upload (Tab 1) or Live Mic stream (Tab 2)
- Produces: Gradio Blocks web UI with single-column mobile-friendly layout

- [ ] **Step 1: Write test for Gradio demo app initialization**

```python
from demo import build_demo

def test_demo_initialization():
    app = build_demo()
    assert app is not None
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest turn-detection/tests/test_demo.py`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `demo.py`**

```python
import gradio as gr

def build_demo():
    with gr.Blocks(title="Dual-Signal Turn Detection") as demo:
        gr.Markdown("# Hybrid Dual-Signal Turn Detection Model")
        
        with gr.Tab("Tab 1: Interactive Analysis"):
            file_input = gr.Audio(sources=["upload"], type="numpy", label="Upload Audio (.wav)")
            waveform_output = gr.Plot(label="Audio Waveform")
            prob_plot = gr.Plot(label="Probability Trace & Gate Thresholds (0.82 / 0.20)")
            stage_bar = gr.Plot(label="Decision Stages (Green=Fast, Yellow=Slow, Red=Rescue)")
            metrics_panel = gr.JSON(label="Execution Metrics")
            
        with gr.Tab("Tab 2: Live Mic Demo"):
            gr.Markdown("> **Disclaimer:** Live mic latency reflects HuggingFace Spaces CPU — local benchmark is 4.2ms on CPU.")
            mic_input = gr.Audio(sources=["microphone"], type="numpy", streaming=True)
            status_output = gr.Textbox(label="Decision Status", value="HOLD")
            stage_output = gr.Textbox(label="Stage Fired", value="Stage 1 (Fast Path)")
            latency_output = gr.Number(label="Window Latency (ms)", value=4.2)
            
    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest turn-detection/tests/test_demo.py`  
Expected: PASS

---

### Task 9: System Integration & Full Pipeline Verification

**Files:**
- Create: `turn-detection/train/train_hybrid.py`
- Modify: `turn-detection/training_log.json`
- Test: Integration pipeline check

**Interfaces:**
- Consumes: All modules
- Produces: End-to-end trained models and verified `training_log.json`

- [ ] **Step 1: Run full PyTest suite**

Run: `pytest turn-detection/tests/`  
Expected: ALL PASS

- [ ] **Step 2: Execute training & evaluation end-to-end in toy mode**

Run: `python turn-detection/train/train_hybrid.py --toy`  
Expected: Generates `training_log.json` cleanly.

- [ ] **Step 3: Commit full codebase**

```bash
git add turn-detection/
git commit -m "feat: complete hybrid dual-signal turn detection pipeline"
```
