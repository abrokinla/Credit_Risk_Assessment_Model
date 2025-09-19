"""
pipeline.py
Defines data processing and modeling pipelines.
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def get_preprocessing_pipeline(numeric_features):
    """Returns a preprocessing pipeline for numeric features."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features)
    ])
    return preprocessor

def get_pipeline(model, numeric_features):
    """Returns a full pipeline with preprocessing and model."""
    preprocessor = get_preprocessing_pipeline(numeric_features)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
