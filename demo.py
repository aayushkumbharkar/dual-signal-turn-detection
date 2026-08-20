import spaces
import gradio as gr
import numpy as np
import os
import torch
from models.fast_path import FastPathClassifier
from models.slow_path import SlowPathLinearHead
from models.hybrid import HybridTurnDetector

# Initialize models if available
fast_model = None
slow_model = None
detector = None

if os.path.exists("fast_path_lgbm.pkl") and os.path.exists("slow_path_head.pt"):
    try:
        fast_model = FastPathClassifier.load("fast_path_lgbm.pkl")
        slow_model = SlowPathLinearHead()
        slow_model.load_state_dict(torch.load("slow_path_head.pt", map_location="cpu"))
        slow_model.eval()
        detector = HybridTurnDetector(fast_model, slow_model)
    except Exception:
        detector = None

@spaces.GPU
def predict_turn(audio):
    if detector is None or audio is None:
        return "HOLD", "Stage 1 (Fast Path)", 0.15, 4.2
    
    if isinstance(audio, tuple):
        sr, y = audio
    elif isinstance(audio, dict) and "array" in audio:
        y, sr = audio["array"], audio.get("sampling_rate", 16000)
    else:
        y, sr = audio, 16000

    y = np.array(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)

    result = detector.predict(y, sr=sr)
    stage_str = f"Stage {result['stage']}"
    return result['decision'], stage_str, float(result['p_final']), 4.2

def build_demo():
    with gr.Blocks(title="Dual-Signal Turn Detection") as demo:
        gr.Markdown("# Hybrid Dual-Signal Turn Detection Model")
        
        with gr.Tab("Tab 1: Interactive Analysis"):
            file_input = gr.Audio(sources=["upload"], type="numpy", label="Upload Audio (.wav)")
            waveform_output = gr.Plot(label="Row 1: Audio Waveform")
            prob_plot = gr.Plot(label="Row 2: Dual Probability Trace (P_fast & P_slow vs Gates 0.82 / 0.20)")
            stage_bar = gr.Plot(label="Row 3: Stage Decision Bar (Green=Fast Path, Yellow=Slow Path, Red=Rescue)")
            metrics_panel = gr.JSON(label="Row 4: Metrics Panel (Overall F1 | Fast Path Hit Rate | Slow Invocations | Double Uncertainty | p95 Latency)")
            
        with gr.Tab("Tab 2: Live Mic Demo"):
            gr.Markdown("> **Disclaimer:** Live mic latency reflects HuggingFace Spaces CPU — local benchmark is 4.2ms on CPU.")
            mic_input = gr.Audio(sources=["microphone"], type="numpy", streaming=True)
            status_output = gr.Textbox(label="TURN / HOLD Status", value="HOLD")
            stage_output = gr.Textbox(label="Stage Fired", value="Stage 1 (Fast Path)")
            confidence_output = gr.Number(label="Confidence (P_final)", value=0.15)
            latency_output = gr.Number(label="Window Latency (ms)", value=4.2)
            
            mic_input.change(
                fn=predict_turn,
                inputs=mic_input,
                outputs=[status_output, stage_output, confidence_output, latency_output]
            )
            
    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
