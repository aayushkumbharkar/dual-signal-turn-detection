import json
import pickle
import argparse
import numpy as np
from data.download_dataset import load_and_split_dataset
from features.acoustic import extract_acoustic_features
from models.fast_path import FastPathClassifier
from models.slow_path import SlowPathModel
from models.hybrid import HybridTurnDetector
from evaluate import evaluate_model

def run_hybrid_training(toy_mode: bool = False):
    print("Step 1: Downloading & splitting dataset...")
    splits, use_pipecat_test, turn_ratio = load_and_split_dataset(toy_mode=toy_mode)
    
    print(f"Step 2: Extracting acoustic features (19 features) for {len(splits['train'])} train samples...")
    X_train_list, y_train_list = [], []
    for sample in splits['train']:
        raw_audio = sample['audio']['array'] if isinstance(sample['audio'], dict) else sample['audio']
        sr = sample['audio']['sampling_rate'] if isinstance(sample['audio'], dict) else 16000
        audio = np.array(raw_audio, dtype=np.float32)
        feats = extract_acoustic_features(audio, sr=sr)
        X_train_list.append(feats)
        label = 1 if sample['label'] in (1, 'TURN') else 0
        y_train_list.append(label)
        
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    
    print("Step 3: Training Fast Path LightGBM classifier...")
    fast_model = FastPathClassifier(n_estimators=100 if toy_mode else 1000)
    fast_model.fit(X_train, y_train)
    
    with open('fast_path_lgbm.pkl', 'wb') as f:
        pickle.dump(fast_model, f)
    
    print("Step 4: Initializing Slow Path model...")
    slow_model = SlowPathModel()
    
    print("Step 5: Assembling Hybrid Turn Detector...")
    hybrid_detector = HybridTurnDetector(fast_model=fast_model, slow_model=slow_model)
    
    print("Step 6: Running evaluation on test split...")
    metrics = evaluate_model(hybrid_detector, splits['test'])
    metrics['test_split_strategy'] = "scenario_a" if use_pipecat_test else "scenario_b"
    metrics['class_balance_ratio'] = turn_ratio
    metrics['optuna_triggered'] = False
    
    with open('training_log.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print("=== Training & Evaluation Complete ===")
    print(json.dumps(metrics, indent=2))
    return hybrid_detector, metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true', help='Run in fast toy mode')
    args = parser.parse_args()
    run_hybrid_training(toy_mode=args.toy)
