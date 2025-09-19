"""
Unit tests for train.py
"""
import pandas as pd
import numpy as np
import os
import tempfile
from src.train import train_model

def test_train_model_basic():
    # Create temporary CSV files
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = os.path.join(tmpdir, 'train.csv')
        test_path = os.path.join(tmpdir, 'test.csv')
        train_df = pd.DataFrame({
            'ApplicantIncome': [1000, 2000],
            'CoapplicantIncome': [500, 0],
            'LoanAmount': [100, 200],
            'Loan_Amount_Term': [360, 360],
            'Dependents': ['0', '1+'],
            'loan_status': [1, 0]
        })
        test_df = pd.DataFrame({
            'ApplicantIncome': [1500],
            'CoapplicantIncome': [300],
            'LoanAmount': [120],
            'Loan_Amount_Term': [360],
            'Dependents': ['2']
        })
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        pipeline = train_model(train_path, test_path, 'logistic_regression', numeric_features, target_col='loan_status')
        preds = pipeline.predict(train_df[numeric_features])
        assert len(preds) == len(train_df)
