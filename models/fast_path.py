import lightgbm as lgb
import numpy as np

class FastPathClassifier:
    def __init__(self, n_estimators=1000, learning_rate=0.05, max_depth=6):
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None):
        if eval_set is not None:
            self.model.fit(
                X, y,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
        else:
            self.model.fit(X, y)
            
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
