import gradio as gr
import numpy as np

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
            
    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
