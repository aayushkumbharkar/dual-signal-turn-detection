import pytest

def test_imports():
    import torch
    import lightgbm
    import librosa
    import numpy
    import datasets
    import transformers
    import gradio
    import gtts
    import scipy
    assert torch.__version__ is not None
    assert lightgbm.__version__ is not None
    assert librosa.__version__ is not None
    assert numpy.__version__ is not None
    assert datasets.__version__ is not None
    assert transformers.__version__ is not None
    assert gradio.__version__ is not None
    assert gtts.__version__ is not None
    assert scipy.__version__ is not None
