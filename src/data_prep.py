"""
data_prep.py
Module for data loading, cleaning, and feature engineering.
"""
import pandas as pd
import numpy as np
def load_data(train_path, test_path):
     pass

def load_data(train_path=None, test_path=None, from_db=False):
    """
    Load train and test data from CSV or database.
    If from_db is True, fetches from the database using load_data_from_db.
    """
    if from_db:
        from db import load_data_from_db
        # Fetch train and test data based on loan_status
        train_df = load_data_from_db(where_clause="loan_status IS NOT NULL")
        test_df = load_data_from_db(where_clause="loan_status IS NULL")
    else:
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
    # First, standardize column names (convert non-underscore to underscore version)
    column_mapping = {
        'applicantincome': 'applicant_income',
        'coapplicantincome': 'coapplicant_income',
        'loanamount': 'loan_amount'
    }
    
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Rename columns if they exist in the non-underscore format
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    # Now proceed with feature engineering using underscore format
    df['total_income'] = df['applicant_income'] + df['coapplicant_income']
    df['loan_repayment_rate'] = df['loan_amount'] / df['loan_amount_term']
    df['loan_amount_ratio'] = df['loan_amount'] / df['applicant_income']
    df['loan_to_income_ratio'] = df['loan_amount'] / df['total_income']
    df['loan_repayment_income_ratio'] = df['loan_repayment_rate'] / df['total_income']
    df['loan_repayment_applicant_income_ratio'] = df['loan_repayment_rate'] / df['applicant_income']
    df['loan_income_thru_term'] = df['applicant_income'] * df['loan_amount_term']
    df['loan_term_income_ratio'] = df['loan_amount_term'] / df['total_income']
    return df

def combine_and_preprocess(train_df, test_df):
    """Combine train and test, preprocess, and feature engineer."""
    train_df['source'] = 'train'
    test_df['source'] = 'test'
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df = col_to_lower_case(combined_df)
    print("[DEBUG] Combined columns after lowercasing:", combined_df.columns.tolist())
    combined_df = preprocess_dependents(combined_df)
    combined_df = feature_eng(combined_df)
    return combined_df
