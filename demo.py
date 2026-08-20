import io
import os
import pickle
import time
import gradio as gr
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is not None:
                return func
            def decorator(f):
                return f
            return decorator

from models.hybrid import HybridTurnDetector
from models.fast_path import FastPathClassifier
from models.slow_path import SlowPathLinearHead

# Module-level model initialization
fast_clf = FastPathClassifier()
if os.path.exists('fast_path_lgbm.pkl'):
    with open('fast_path_lgbm.pkl', 'rb') as f:
        obj = pickle.load(f)
        if isinstance(obj, FastPathClassifier):
            fast_clf = obj
        else:
            fast_clf.model = obj

slow_head = SlowPathLinearHead()
if os.path.exists('slow_path_head.pt'):
    slow_head.load_state_dict(torch.load('slow_path_head.pt', map_location='cpu'))
    slow_head.eval()

detector = HybridTurnDetector(
    fast_model=fast_clf,
    slow_model=slow_head
)

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img

@spaces.GPU
def analyze_audio_file(audio_input):
    if audio_input is None:
        return None, None, "No Audio Provided", {}
    
    if isinstance(audio_input, tuple):
        sr, y = audio_input
    elif isinstance(audio_input, dict):
        sr = audio_input.get("sampling_rate", 16000)
        y = audio_input.get("array", None)
        if y is None:
            return None, None, "Invalid Audio Format", {}
    else:
        return None, None, "Invalid Audio Format", {}

    if y is None or len(y) == 0:
        return None, None, "Empty Audio", {}

    y = y.astype(np.float32)
    if np.abs(y).max() > 1.0:
        y = y / 32768.0

    if y.ndim > 1:
        y = y.mean(axis=1)
    
    # Run hybrid detector
    result = detector.predict(y, buffer_dur=0.0, sr=sr)
    
    # Generate waveform plot
    fig1, ax1 = plt.subplots(figsize=(10, 2))
    t = np.linspace(0, len(y)/sr, len(y))
    ax1.plot(t, y, color='#4CAF50', linewidth=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Audio Waveform')
    plt.tight_layout()
    img1 = fig_to_pil(fig1)
    
    # Generate probability trace plot
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.axhline(y=0.82, color='red', linestyle='--', label='TURN_GATE=0.82')
    ax2.axhline(y=0.20, color='blue', linestyle='--', label='HOLD_GATE=0.20')
    ax2.plot([0, 1], [result['p_fast'], result['p_fast']], color='gray', linewidth=1.5, label=f"P_fast={result['p_fast']:.4f}")
    if result['p_slow'] is not None:
        ax2.plot([0, 1], [result['p_slow'], result['p_slow']], color='orange', linewidth=2, label=f"P_slow={result['p_slow']:.4f}")
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')
    ax2.set_title(f"Probability Trace (Decision: {result['decision']})")
    plt.tight_layout()
    img2 = fig_to_pil(fig2)
    
    stage_labels = {
        0: 'Stage 0 — Max Buffer Exceeded',
        1: 'Stage 1 — Fast Path (LightGBM only)', 
        2: 'Stage 2 — Slow Path (Whisper invoked)',
        3: 'Stage 3 — Double Uncertainty Rescue'
    }
    
    stage_name = stage_labels.get(result['stage'], f"Stage {result['stage']}")
    
    metrics = {
        'decision': result['decision'],
        'stage': stage_name,
        'p_fast': round(result['p_fast'], 4),
        'p_slow': round(result['p_slow'], 4) if result['p_slow'] is not None else 'N/A',
        'p_final': round(result['p_final'], 4),
        'double_uncertainty_count': detector.double_uncertainty_count
    }
    
    return img1, img2, stage_name, metrics

@spaces.GPU  
def analyze_mic_input(audio_input):
    if audio_input is None:
        return "Waiting for mic input...", "No Stage Fired", 0.0
    
    if isinstance(audio_input, tuple):
        sr, y = audio_input
    elif isinstance(audio_input, dict):
        sr = audio_input.get("sampling_rate", 16000)
        y = audio_input.get("array", None)
        if y is None:
            return "Waiting for mic input...", "No Stage Fired", 0.0
    else:
        return "Waiting for mic input...", "No Stage Fired", 0.0

    if y is None or len(y) == 0:
        return "Waiting for mic input...", "No Stage Fired", 0.0

    y = y.astype(np.float32)
    if np.abs(y).max() > 1.0:
        y = y / 32768.0

    if y.ndim > 1:
        y = y.mean(axis=1)
    
    t0 = time.perf_counter()
    result = detector.predict(y, buffer_dur=0.0, sr=sr)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    stage_labels = {
        0: 'Stage 0 — Max Buffer Exceeded',
        1: 'Stage 1 — Fast Path',
        2: 'Stage 2 — Slow Path', 
        3: 'Stage 3 — Rescue'
    }
    
    return (
        result['decision'],
        stage_labels.get(result['stage'], f"Stage {result['stage']}"),
        round(latency_ms, 2)
    )

def build_demo():
    with gr.Blocks(title="Dual-Signal Turn Detection") as demo:
        gr.Markdown("# Hybrid Dual-Signal Turn Detection Model")
        
        with gr.Tab("Tab 1: Interactive Analysis"):
            file_input = gr.Audio(sources=["upload"], type="numpy", label="Upload Audio (.wav)")
            analyze_btn = gr.Button("Analyze Audio", variant="primary")
            waveform_output = gr.Image(label="Row 1: Audio Waveform")
            prob_plot = gr.Image(label="Row 2: Dual Probability Trace (P_fast & P_slow vs Gates 0.82 / 0.20)")
            stage_bar = gr.Textbox(label="Row 3: Stage Decision Bar")
            metrics_panel = gr.JSON(label="Row 4: Metrics Panel")

            file_outputs = [waveform_output, prob_plot, stage_bar, metrics_panel]

            analyze_btn.click(
                fn=analyze_audio_file,
                inputs=[file_input],
                outputs=file_outputs
            )
            file_input.upload(
                fn=analyze_audio_file,
                inputs=[file_input],
                outputs=file_outputs
            )
            file_input.change(
                fn=analyze_audio_file,
                inputs=[file_input],
                outputs=file_outputs
            )
            
        with gr.Tab("Tab 2: Live Mic Demo"):
            gr.Markdown("> **Disclaimer:** Live mic latency reflects HuggingFace Spaces CPU — local benchmark is 4.2ms on CPU.")
            mic_input = gr.Audio(sources=["microphone"], type="numpy", label="Record Microphone Audio")
            analyze_mic_btn = gr.Button("Analyze Microphone Audio", variant="primary")
            status_output = gr.Textbox(label="TURN / HOLD Status", value="Waiting for mic input...")
            stage_output = gr.Textbox(label="Stage Fired", value="No Stage Fired")
            latency_output = gr.Number(label="Window Latency (ms)", value=0.0)
            
            mic_outputs = [status_output, stage_output, latency_output]

            analyze_mic_btn.click(
                fn=analyze_mic_input,
                inputs=[mic_input],
                outputs=mic_outputs
            )
            mic_input.stop_recording(
                fn=analyze_mic_input,
                inputs=[mic_input],
                outputs=mic_outputs
            )
            mic_input.change(
                fn=analyze_mic_input,
                inputs=[mic_input],
                outputs=mic_outputs
            )
            
    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
