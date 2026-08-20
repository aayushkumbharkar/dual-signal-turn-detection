import librosa
import numpy as np

def apply_pitch_shift(y: np.ndarray, sr: int = 16000, n_steps: float = 2.0) -> np.ndarray:
    if len(y) == 0:
        return y
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

def inject_gtts_filler(y: np.ndarray, sr: int = 16000, filler_text: str = "acha") -> np.ndarray:
    # Synthetic filler injection placeholder returning padded audio array
    filler_audio = np.random.randn(sr // 2).astype(np.float32) * 0.1
    return np.concatenate((y, filler_audio))
