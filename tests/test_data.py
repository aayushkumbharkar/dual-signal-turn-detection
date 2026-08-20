import pytest
from data.download_dataset import load_and_split_dataset

def test_load_and_split_dataset():
    splits, use_pipecat_test, turn_ratio = load_and_split_dataset(toy_mode=True)
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits
    assert 0.0 <= turn_ratio <= 1.0
    assert isinstance(use_pipecat_test, bool)
