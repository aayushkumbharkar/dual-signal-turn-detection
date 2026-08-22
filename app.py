try:
    import spaces
except ImportError:
    pass
import gradio as gr
from demo import build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", show_error=True)
