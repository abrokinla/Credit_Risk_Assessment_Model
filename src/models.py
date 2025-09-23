"""
models.py
Defines ML and DL models (scikit-learn, XGBoost, CatBoost, PyTorch).
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# PyTorch model for advanced use
try:
    import torch
    import torch.nn as nn
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)
except ImportError:
    SimpleMLP = None


# Ensemble model for averaging predictions
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class EnsembleAverager(BaseEstimator, ClassifierMixin):
    """
    Ensemble model that averages the predicted probabilities of multiple fitted models.
    """
    def __init__(self, models=None):
        self.models = models
        self._is_fitted = False

    def fit(self, X, y):
        """
        No fitting required for this meta-estimator as it uses pre-fitted models.
        """
        if self.models is None:
            raise ValueError("Models list cannot be None")
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        if not self._is_fitted:
            raise ValueError("Estimator not fitted, call 'fit' before making predictions")
        probas = [m.predict_proba(X)[:, 1] for m in self.models]
        avg_proba = np.mean(probas, axis=0)
        # Return shape (n_samples, 2) for sklearn compatibility
        return np.vstack([1 - avg_proba, avg_proba]).T

    def predict(self, X):
        if not self._is_fitted:
            raise ValueError("Estimator not fitted, call 'fit' before making predictions")
        avg_proba = self.predict_proba(X)[:, 1]
        return (avg_proba >= 0.5).astype(int)

def get_model(model_name, **kwargs):
    """Factory for ML and DL models."""
    if model_name == "logistic_regression":
        return LogisticRegression(class_weight="balanced", random_state=42, **kwargs)
    elif model_name == "random_forest":
        return RandomForestClassifier(class_weight="balanced", random_state=42, **kwargs)
    elif model_name == "xgboost":
        return XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss", **kwargs)
    elif model_name == "catboost":
        return CatBoostClassifier(verbose=False, random_state=42, **kwargs)
    elif model_name == "pytorch_mlp":
        if SimpleMLP is None:
            raise ImportError("PyTorch is not installed.")
        input_dim = kwargs.get('input_dim', 10)
        hidden_dim = kwargs.get('hidden_dim', 64)
        dropout = kwargs.get('dropout', 0.2)
        return SimpleMLP(input_dim, hidden_dim, dropout)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

