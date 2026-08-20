import torch
import torch.nn as nn
import numpy as np

def pool_trailing_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    # hidden_states shape: (Batch, Time, 384)
    T = hidden_states.shape[1]
    split_idx = int(0.75 * T)
    trailing_states = hidden_states[:, split_idx:, :]
    return torch.mean(trailing_states, dim=1)

class SlowPathLinearHead(nn.Module):
    def __init__(self, in_features: int = 384, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SlowPathModel:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.head = SlowPathLinearHead().to(device)
        self.head.eval()
        
    def predict(self, y: np.ndarray) -> float:
        # Mock / Fast dummy inference for slow path when Whisper model is offline/dummy
        # Generate deterministic synthetic pooled embedding from input audio array
        with torch.no_grad():
            emb = torch.from_numpy(np.zeros((1, 384), dtype=np.float32)).to(self.device)
            p = float(self.head(emb).item())
        return p
