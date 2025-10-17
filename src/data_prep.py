"""
data_prep.py
Module for data loading, cleaning, and feature engineering.
"""
import pandas as pd
import numpy as np

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

        # Ensure all expected dummies are present, even if some categories are missing
        expected_dummies = ['dependents_0', 'dependents_1', 'dependents_2', 'dependents_3']
        for col in expected_dummies:
            if col not in df.columns:
                df[col] = 0
    return df

def feature_eng(df):
    """Feature engineering as in the notebook."""
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Ensure we have the required columns
    required_cols = ['applicant_income', 'coapplicant_income', 'loan_amount', 'loan_amount_term']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"[WARNING] Missing columns for feature engineering: {missing_cols}")
        print(f"[DEBUG] Available columns: {list(df.columns)}")
        return df
    
    # Feature engineering
    df['total_income'] = df['applicant_income'] + df['coapplicant_income']
    df['loan_repayment_rate'] = df['loan_amount'] / df['loan_amount_term']
    df['loan_amount_ratio'] = df['loan_amount'] / df['applicant_income'].replace(0, np.nan)
    df['loan_to_income_ratio'] = df['loan_amount'] / df['total_income'].replace(0, np.nan)
    df['loan_repayment_income_ratio'] = df['loan_repayment_rate'] / df['total_income'].replace(0, np.nan)
    df['loan_repayment_applicant_income_ratio'] = df['loan_repayment_rate'] / df['applicant_income'].replace(0, np.nan)
    df['loan_income_thru_term'] = df['applicant_income'] * df['loan_amount_term']
    df['loan_term_income_ratio'] = df['loan_amount_term'] / df['total_income'].replace(0, np.nan)
    
    return df

def combine_and_preprocess(train_df, test_df):
    """
    Combine train and test data, preprocess, and feature engineer.
    Adds a 'source' column to distinguish datasets,
    ensures lowercase consistency, and handles feature engineering.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df['source'] = 'train'
    test_df['source'] = 'test'
    
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df = col_to_lower_case(combined_df)
    combined_df['source'] = combined_df['source'].str.lower()

    print(f"[DEBUG] Combined shape: {combined_df.shape}")
    print(f"[DEBUG] Columns before preprocessing: {list(combined_df.columns)}")

    combined_df = preprocess_dependents(combined_df)
    combined_df = feature_eng(combined_df)
    
    print(f"[DEBUG] Columns after preprocessing: {list(combined_df.columns)}")
    
    return combined_df
