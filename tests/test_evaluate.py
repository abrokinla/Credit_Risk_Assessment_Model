"""
Unit tests for evaluate.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.evaluate import evaluate_classification_model

def test_evaluate_classification_model():
    # Simple binary classification example
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression().fit(X, y)
    results = evaluate_classification_model(model, X, y)
    assert 'accuracy' in results
    assert 'precision' in results
    assert 'recall' in results
    assert 'f1' in results
    assert 'confusion_matrix' in results
    assert 'classification_report' in results
    assert 'roc_auc' in results
    assert results['accuracy'] >= 0
