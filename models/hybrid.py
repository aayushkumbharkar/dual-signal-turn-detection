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
