---
title: Dual-Signal Turn Detection
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.16.0
app_file: app.py
pinned: false
license: mit
short_description: Low-latency hybrid turn detection for Hinglish voice AI.
---

# Dual-Signal Turn Detection Model

This HuggingFace Space hosts the interactive demo for the **Dual-Signal Turn Detection Model** designed for low-latency conversational voice AI.

> **Note on Latency**: Live mic latency on HuggingFace Spaces reflects Spaces CPU execution — local benchmark is **8.3ms** on CPU.

## Architecture
- **Fast Path (Stage 1)**: 19 acoustic features extracted via `librosa.yin` (<3ms) + LightGBM classifier.
- **Slow Path (Stage 2)**: Frozen `openai/whisper-tiny` encoder + trailing pooling + 2-layer MLP head.
- **Cascaded Gating**: `TURN_GATE = 0.82`, `HOLD_GATE = 0.20`, Stage 3 Double-Uncertainty Fast Path rescue.
