import pickle
import numpy as np
from sklearn.metrics import f1_score
from data.download_dataset import load_and_split_dataset
from features.acoustic import extract_acoustic_features
from models.fast_path import FastPathClassifier

def run_train_fast(toy_mode=False):
    print("=== Step 3: Training Fast Path LightGBM Classifier ===")
    splits, _, turn_ratio = load_and_split_dataset(toy_mode=toy_mode)
    
    print(f"Extracting features for {len(splits['train'])} train samples...")
    X_train_list, y_train_list = [], []
    for s in splits['train']:
        raw_audio = s['audio']['array'] if isinstance(s['audio'], dict) else s['audio']
        sr = s['audio']['sampling_rate'] if isinstance(s['audio'], dict) else 16000
        feats = extract_acoustic_features(np.array(raw_audio, dtype=np.float32), sr=sr)
        X_train_list.append(feats)
        y_train_list.append(1 if s['label'] in (1, 'TURN') else 0)
        
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    
    print(f"Extracting features for {len(splits['val'])} val samples...")
    X_val_list, y_val_list = [], []
    for s in splits['val']:
        raw_audio = s['audio']['array'] if isinstance(s['audio'], dict) else s['audio']
        sr = s['audio']['sampling_rate'] if isinstance(s['audio'], dict) else 16000
        feats = extract_acoustic_features(np.array(raw_audio, dtype=np.float32), sr=sr)
        X_val_list.append(feats)
        y_val_list.append(1 if s['label'] in (1, 'TURN') else 0)
        
    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.int32)
    
    clf = FastPathClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6)
    clf.fit(X_train, y_train, eval_X=X_val, eval_y=y_val)
    
    val_probs = clf.predict_proba(X_val)
    val_preds = (val_probs >= 0.50).astype(int)
    val_f1 = float(f1_score(y_val, val_preds, zero_division=0))
    best_iter = getattr(clf.model, 'best_iteration_', 1000)
    
    print(f"\n--- Fast Path Results ---")
    print(f"Validation F1: {val_f1:.4f}")
    print(f"Best Iteration: {best_iter}")
    
    optuna_triggered = False
    if val_f1 < 0.84:
        print("Validation F1 < 0.84 — Optuna hyperparameter sweep triggered!")
        optuna_triggered = True
    else:
        print("Validation F1 >= 0.84 — Default parameters optimal, Optuna skipped.")
        
    with open('fast_path_lgbm.pkl', 'wb') as f:
        pickle.dump(clf, f)
        
    return clf, val_f1, best_iter, optuna_triggered

if __name__ == '__main__':
    run_train_fast(toy_mode=False)
