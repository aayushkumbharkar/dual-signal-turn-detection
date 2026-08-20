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
