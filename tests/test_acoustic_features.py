import numpy as np
import time
from features.acoustic import extract_acoustic_features

def test_extract_acoustic_features_shape_and_latency():
    sr = 16000
    # Generate 1.5s sine wave (440Hz tone) which pyin tracks fast
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    # Warmup call
    _ = extract_acoustic_features(y[:1000], sr)
    
    t0 = time.perf_counter()
    feats = extract_acoustic_features(y, sr)
    t1 = time.perf_counter()
    
    assert feats.shape == (19,)
    assert not np.isnan(feats).any()
    assert (t1 - t0) * 1000.0 < 1000.0  # reasonable latency check
