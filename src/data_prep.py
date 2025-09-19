"""
data_prep.py
Module for data loading, cleaning, and feature engineering.
"""
def load_data(train_path, test_path):
import pandas as pd
import numpy as np

def load_data(train_path, test_path):
    """Load train and test CSV files."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def col_to_lower_case(df):
    """Convert all column names to lower case."""
    df.columns = [col.lower() for col in df.columns]
    return df

def preprocess_dependents(df):
    """Clean and one-hot encode the 'dependents' column."""
    if 'dependents' in df.columns:
        df['dependents'] = df['dependents'].astype(str).str.replace('+', '', regex=False)
        dependents_dummies = pd.get_dummies(df['dependents'], prefix='dependents')
        df = pd.concat([df, dependents_dummies], axis=1)
        df = df.drop('dependents', axis=1)
    return df

def feature_eng(df):
    """Feature engineering as in the notebook."""
    df['total_income'] = df['applicantincome'] + df['coapplicantincome']
    df['loan_repayment_rate'] = df['loanamount'] / df['loan_amount_term']
    df['loan_amount_ratio'] = df['loanamount'] / df['applicantincome']
    df['loan_to_income_ratio'] = df['loanamount'] / df['total_income']
    df['loan_repayment_income_ratio'] = df['loan_repayment_rate'] / df['total_income']
    df['loan_repayment_applicatnt_income_ratio'] = df['loan_repayment_rate'] / df['applicantincome']
    df['loan_income_thru_term'] = df['applicantincome'] * df['loan_amount_term']
    df['loan_term_income_ratio'] = df['loan_amount_term'] / df['total_income']
    return df

def combine_and_preprocess(train_df, test_df):
    """Combine train and test, preprocess, and feature engineer."""
    train_df['source'] = 'train'
    test_df['source'] = 'test'
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df = col_to_lower_case(combined_df)
    combined_df = preprocess_dependents(combined_df)
    combined_df = feature_eng(combined_df)
    return combined_df
