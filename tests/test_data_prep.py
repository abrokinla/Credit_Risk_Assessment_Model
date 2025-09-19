"""
Unit tests for data_prep.py
"""

import pandas as pd
import numpy as np
import os
import tempfile
import pytest
from src import data_prep

def test_load_data():
    # Create temporary CSV files
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = os.path.join(tmpdir, 'train.csv')
        test_path = os.path.join(tmpdir, 'test.csv')
        train_df = pd.DataFrame({
            'ApplicantIncome': [1000, 2000],
            'CoapplicantIncome': [500, 0],
            'LoanAmount': [100, 200],
            'Loan_Amount_Term': [360, 360],
            'Dependents': ['0', '1+']
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

        loaded_train, loaded_test = data_prep.load_data(train_path, test_path)
        assert loaded_train.shape == (2, 5)
        assert loaded_test.shape == (1, 5)

def test_col_to_lower_case():
    df = pd.DataFrame({'A': [1], 'B': [2]})
    df = data_prep.col_to_lower_case(df)
    assert list(df.columns) == ['a', 'b']

def test_preprocess_dependents():
    df = pd.DataFrame({'dependents': ['0', '1', '2', '3+']})
    df = data_prep.preprocess_dependents(df)
    assert 'dependents_3' in df.columns
    assert 'dependents_3+' not in df.columns

def test_feature_eng():
    df = pd.DataFrame({
        'applicantincome': [1000],
        'coapplicantincome': [500],
        'loanamount': [100],
        'loan_amount_term': [360]
    })
    df = data_prep.feature_eng(df)
    assert 'total_income' in df.columns
    assert np.isclose(df['total_income'][0], 1500)
