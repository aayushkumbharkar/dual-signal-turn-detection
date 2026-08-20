import spaces
import gradio as gr
from demo import build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch()
