import pytest
import numpy as np
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
