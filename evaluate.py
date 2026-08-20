import json
import time
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from data.download_dataset import load_and_split_dataset
from models.fast_path import FastPathClassifier
from models.slow_path import SlowPathModel
from models.hybrid import HybridTurnDetector

def evaluate_model(hybrid_detector, test_dataset) -> dict:
    predictions = []
    labels = []
    latencies = []
    fast_hits = 0
    slow_invocations = 0
    
    for sample in test_dataset:
        raw_audio = sample['audio']['array'] if isinstance(sample['audio'], dict) else sample['audio']
        sr = sample['audio']['sampling_rate'] if isinstance(sample['audio'], dict) else 16000
        audio = np.array(raw_audio, dtype=np.float32)
        true_label = 1 if sample['label'] in (1, 'TURN') else 0
        
        t0 = time.perf_counter()
        result = hybrid_detector.predict(audio, buffer_dur=0.0, sr=sr)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        
        latencies.append(lat_ms)
        pred_label = 1 if result['decision'] == 'TURN' else 0
        predictions.append(pred_label)
        labels.append(true_label)
        
        if result['stage'] == 1:
            fast_hits += 1
        elif result['stage'] in (2, 3):
            slow_invocations += 1
            
    total = len(test_dataset) if len(test_dataset) > 0 else 1
    f1 = float(f1_score(labels, predictions, zero_division=0))
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
    results = {
        'overall_f1': f1,
        'fpr': fpr,
        'fnr': fnr,
        'fast_path_hit_rate': float(fast_hits / total),
        'slow_path_invocation_rate': float(slow_invocations / total),
        'double_uncertainty_count': hybrid_detector.double_uncertainty_count,
        'p50_latency_ms': float(np.percentile(latencies, 50)) if len(latencies) > 0 else 0.0,
        'p95_latency_ms': float(np.percentile(latencies, 95)) if len(latencies) > 0 else 0.0
    }
    
    with open('training_log.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == '__main__':
    print("=== Step 6: Final Model Evaluation ===")
    splits, use_pipecat_test, turn_ratio = load_and_split_dataset(toy_mode=False)
    
    # Load fast model and slow model
    try:
        import pickle
        with open('fast_path_lgbm.pkl', 'rb') as f:
            fast_model = pickle.load(f)
    except Exception:
        fast_model = FastPathClassifier()
        
    slow_model = SlowPathModel()
    hybrid_detector = HybridTurnDetector(fast_model, slow_model)
    
    metrics = evaluate_model(hybrid_detector, splits['test'])
    
    print("\n--- Final Evaluation Metrics ---")
    print(f"Overall F1 Score:           {metrics['overall_f1']:.4f}")
    print(f"False Positive Rate (FPR):   {metrics['fpr']:.4f}")
    print(f"False Negative Rate (FNR):   {metrics['fnr']:.4f}")
    print(f"Fast Path Hit Rate:         {metrics['fast_path_hit_rate']*100:.2f}%")
    print(f"Slow Path Invocation Rate:  {metrics['slow_path_invocation_rate']*100:.2f}%")
    print(f"Double Uncertainty Count:   {metrics['double_uncertainty_count']}")
    print(f"Latency p50:                {metrics['p50_latency_ms']:.2f} ms")
    print(f"Latency p95:                {metrics['p95_latency_ms']:.2f} ms")
