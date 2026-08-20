import os
import numpy as np
import librosa
from data.download_dataset import load_and_split_dataset

def run_eda(toy_mode=False):
    print("=== Step 2: Running Exploratory Data Analysis (EDA) ===")
    splits, use_pipecat_test, turn_ratio = load_and_split_dataset(toy_mode=toy_mode)
    train_ds = splits['train']
    
    durations = []
    silence_ratios = []
    filler_counts = 0
    total_text_samples = 0
    
    filler_words = {"basically", "right", "so", "okay", "like", "um", "uh", "acha", "matlab", "toh"}
    
    for sample in train_ds:
        audio = sample['audio']['array'] if isinstance(sample['audio'], dict) else sample['audio']
        sr = sample['audio']['sampling_rate'] if isinstance(sample['audio'], dict) else 16000
        y = np.array(audio, dtype=np.float32)
        
        dur = len(y) / sr
        durations.append(dur)
        
        rms = librosa.feature.rms(y=y)[0] if len(y) > 0 else np.array([0])
        rms_mean = np.mean(rms)
        is_silent = rms < (0.01 * rms_mean + 1e-6)
        silence_ratios.append(float(np.mean(is_silent)))
        
        text = sample.get('text', '') or sample.get('transcript', '')
        if text:
            total_text_samples += 1
            words = set(text.lower().split())
            if words.intersection(filler_words):
                filler_counts += 1
                
    avg_dur = float(np.mean(durations)) if durations else 0.0
    p50_dur = float(np.median(durations)) if durations else 0.0
    avg_silence = float(np.mean(silence_ratios)) if silence_ratios else 0.0
    p50_silence = float(np.median(silence_ratios)) if silence_ratios else 0.0
    filler_prevalence = float(filler_counts / total_text_samples) if total_text_samples > 0 else 0.0
    
    report_md = f"""# Exploratory Data Analysis (EDA) Report

**Dataset Split Strategy**: {'Scenario A (official test set)' if use_pipecat_test else 'Scenario B (stratified 70/15/15)'}  
**Train Samples**: {len(splits['train'])}  
**Val Samples**: {len(splits['val'])}  
**Test Samples**: {len(splits['test'])}  

## Key Statistics
- **Class Balance Ratio (TURN)**: {turn_ratio:.4f} ({turn_ratio*100:.2f}%)
- **Average Audio Duration**: {avg_dur:.2f}s (Median p50: {p50_dur:.2f}s)
- **Silence Ratio Distribution**: Mean {avg_silence*100:.2f}%, Median {p50_silence*100:.2f}%
- **Filler Word Prevalence**: {filler_prevalence*100:.2f}% of samples ({filler_counts}/{total_text_samples})
"""

    with open('eda_report.md', 'w') as f:
        f.write(report_md)
        
    print(report_md)
    return report_md

if __name__ == '__main__':
    run_eda(toy_mode=False)
