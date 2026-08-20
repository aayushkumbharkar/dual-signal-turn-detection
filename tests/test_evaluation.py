import numpy as np
from data.augment import apply_pitch_shift
from evaluate import evaluate_model
from models.hybrid import HybridTurnDetector

class DummyFastPath:
    def predict_proba(self, X): return np.array([0.85])

class DummySlowPath:
    def predict(self, y): return 0.85

def test_hinglish_augmentation():
    y = np.random.randn(16000).astype(np.float32)
    y_shifted = apply_pitch_shift(y, sr=16000, n_steps=2.0)
    assert len(y_shifted) > 0
    assert not np.array_equal(y, y_shifted)

def test_evaluate_model():
    detector = HybridTurnDetector(DummyFastPath(), DummySlowPath())
    test_ds = [
        {'audio': np.zeros(16000, dtype=np.float32), 'label': 1},
        {'audio': np.zeros(16000, dtype=np.float32), 'label': 0}
    ]
    res = evaluate_model(detector, test_ds)
    assert 'overall_f1' in res
    assert 'fpr' in res
    assert 'fnr' in res
    assert 'p50_latency_ms' in res
    assert 'p95_latency_ms' in res
