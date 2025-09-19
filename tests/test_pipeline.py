"""
Unit tests for pipeline.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.pipeline import get_pipeline

def test_get_pipeline_fit_predict():
    # Create a simple DataFrame
    X = pd.DataFrame({
        'num1': [1, 2, 3, np.nan],
        'num2': [4, 5, 6, 7]
    })
    y = [0, 1, 0, 1]
    numeric_features = ['num1', 'num2']
    model = LogisticRegression()
    pipeline = get_pipeline(model, numeric_features)
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(y)
    # Check that pipeline handles missing values
    assert not np.isnan(pipeline.named_steps['preprocessor'].transform(X)).any()
