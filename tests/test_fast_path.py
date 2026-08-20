import numpy as np
from models.fast_path import FastPathClassifier

def test_fast_path_fit_predict():
    X = np.random.randn(100, 19).astype(np.float32)
    y = np.random.randint(0, 2, size=(100,))
    
    clf = FastPathClassifier()
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    
    assert probs.shape == (100,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
