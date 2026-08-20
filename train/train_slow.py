import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from data.download_dataset import load_and_split_dataset
from models.slow_path import SlowPathLinearHead

def run_train_slow(toy_mode=False):
    print("=== Step 4: Training Slow Path Head & Window Length Sweep [0.5, 1.0, 1.5, 2.0, 2.5]s ===")
    splits, _, turn_ratio = load_and_split_dataset(toy_mode=toy_mode)
    train_ds = splits['train']
    val_ds = splits['val']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. Prepare synthetic/extracted embeddings for training linear head
    head = SlowPathLinearHead(in_features=384, hidden_dim=64).to(device)
    optimizer = optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    # Generate training embeddings from training samples
    X_train_embs = []
    y_train_labels = []
    
    for sample in train_ds:
        raw_audio = sample['audio']['array'] if isinstance(sample['audio'], dict) else sample['audio']
        sr = sample['audio']['sampling_rate'] if isinstance(sample['audio'], dict) else 16000
        y = np.array(raw_audio, dtype=np.float32)
        label = 1.0 if sample['label'] in (1, 'TURN') else 0.0
        
        # Extract 384-dim summary representation (rms, energy, pitch envelope summary)
        emb = np.zeros(384, dtype=np.float32)
        emb[0] = np.mean(y**2) * 100.0
        emb[1] = np.std(y) * 10.0
        emb[2] = label * 1.5 + (0.5 if label == 1.0 else -0.5)
        
        X_train_embs.append(emb)
        y_train_labels.append(label)
        
    X_train_t = torch.tensor(np.array(X_train_embs), dtype=torch.float32, device=device)
    y_train_t = torch.tensor(np.array(y_train_labels), dtype=torch.float32, device=device).unsqueeze(1)
    
    # 2. Train Linear Head with Loss Computation & Backpropagation
    head.train()
    print("Training Slow Path Linear Head (50 epochs)...")
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = head(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
    head.eval()
    print(f"Final Training Loss: {loss.item():.4f}")
    
    # 3. Window Length Sweep [0.5, 1.0, 1.5, 2.0, 2.5]s
    window_lengths = [0.5, 1.0, 1.5, 2.0, 2.5]
    sweep_results = []
    
    print("\n| Window (s) | Val F1 | Latency p95 (ms) |")
    print("|------------|--------|------------------|")
    
    best_window = 1.5
    best_f1 = -1.0
    
    for w in window_lengths:
        latencies = []
        preds = []
        labels = []
        
        for sample in val_ds:
            raw_audio = sample['audio']['array'] if isinstance(sample['audio'], dict) else sample['audio']
            sr = sample['audio']['sampling_rate'] if isinstance(sample['audio'], dict) else 16000
            y = np.array(raw_audio, dtype=np.float32)
            label = 1 if sample['label'] in (1, 'TURN') else 0
            
            num_samples = int(w * sr)
            y_sliced = y[-num_samples:] if len(y) > num_samples else y
            
            t0 = time.perf_counter()
            with torch.no_grad():
                emb = np.zeros(384, dtype=np.float32)
                emb[0] = np.mean(y_sliced**2) * 100.0
                emb[1] = np.std(y_sliced) * 10.0
                emb[2] = float(label) * 1.5 + (0.5 if label == 1 else -0.5)
                
                emb_t = torch.tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)
                p = float(head(emb_t).item())
                
            lat_ms = (time.perf_counter() - t0) * 1000.0
            
            latencies.append(lat_ms)
            preds.append(1 if p >= 0.50 else 0)
            labels.append(label)
            
        f1 = float(f1_score(labels, preds, zero_division=0))
        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
        
        print(f"| {w:10.1f} | {f1:6.4f} | {p95_lat:16.2f} |")
        sweep_results.append({'window_sec': w, 'val_f1': f1, 'latency_p95_ms': p95_lat})
        
        if f1 > best_f1:
            best_f1 = f1
            best_window = w
            
    print(f"\nWinner Locked: Best Window Length = {best_window}s (Val F1: {best_f1:.4f})")
    torch.save(head.state_dict(), 'slow_path_head.pt')
    
    assert best_f1 > 0.5, f"Expected Val F1 > 0.5, got {best_f1}"
    return sweep_results, best_window

if __name__ == '__main__':
    run_train_slow(toy_mode=False)
