import numpy as np
import librosa

def extract_acoustic_features(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    if len(y) == 0:
        return np.zeros(19, dtype=np.float32)
    
    hop_length = 512
    frame_length = 1024
    
    # 1. Fast F0 via YIN (6 features) - <3ms latency budget
    try:
        f0 = librosa.yin(y, fmin=80, fmax=400, sr=sr, frame_length=frame_length, hop_length=hop_length)
        voiced_f0 = f0[(f0 > 82.0) & (f0 < 398.0)]
    except Exception:
        voiced_f0 = np.array([])
        f0 = np.array([])
        
    if len(voiced_f0) > 0:
        f0_mean = float(np.mean(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        f0_slope = float(np.polyfit(np.arange(len(voiced_f0)), voiced_f0, 1)[0]) if len(voiced_f0) > 1 else 0.0
        f0_final_frame = float(voiced_f0[-1])
        voiced_frame_ratio = float(len(voiced_f0) / max(1, len(f0)))
    else:
        f0_mean, f0_std, f0_slope, f0_final_frame, voiced_frame_ratio = 0.0, 0.0, 0.0, 0.0, 0.0
    
    f0_final_minus_mean = f0_final_frame - f0_mean

    # 2. Energy / RMS (6 features)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    rms_slope = float(np.polyfit(np.arange(len(rms)), rms, 1)[0]) if len(rms) > 1 else 0.0
    rms_final_frame = float(rms[-1]) if len(rms) > 0 else 0.0
    
    split_idx = int(0.75 * len(rms))
    head_rms = np.mean(rms[:split_idx]) if split_idx > 0 else 1e-6
    tail_rms = np.mean(rms[split_idx:]) if split_idx < len(rms) else 0.0
    rms_tail_ratio = float(tail_rms / (head_rms + 1e-6))
    rms_final_minus_mean = rms_final_frame - rms_mean

    # 3. Silence (4 features)
    silence_thresh = 0.01 * (rms_mean + 1e-6)
    is_silent = rms < silence_thresh
    silence_ratio = float(np.mean(is_silent))
    
    trailing_count = 0
    for s in reversed(is_silent):
        if s: trailing_count += 1
        else: break
    frame_dur = hop_length / sr
    trailing_silence_duration = float(trailing_count * frame_dur)
    
    if len(is_silent) > 0:
        padded = np.concatenate(([False], is_silent, [False]))
        diffs = np.diff(padded.astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        runs = ends - starts
        longest_silence_run = float(np.max(runs)) if len(runs) > 0 else 0.0
        silence_threshold_crossings = float(np.sum(np.diff(is_silent.astype(int)) != 0))
    else:
        longest_silence_run = 0.0
        silence_threshold_crossings = 0.0

    # 4. Spectral (3 features)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_centroid_mean = float(np.mean(spec_cent))
    spectral_centroid_slope = float(np.polyfit(np.arange(len(spec_cent)), spec_cent, 1)[0]) if len(spec_cent) > 1 else 0.0
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]
    zcr_mean = float(np.mean(zcr))

    return np.array([
        f0_mean, f0_std, f0_slope, f0_final_frame, voiced_frame_ratio, f0_final_minus_mean,
        rms_mean, rms_std, rms_slope, rms_final_frame, rms_tail_ratio, rms_final_minus_mean,
        trailing_silence_duration, silence_ratio, longest_silence_run, silence_threshold_crossings,
        spectral_centroid_mean, spectral_centroid_slope, zcr_mean
    ], dtype=np.float32)
