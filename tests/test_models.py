"""
Unit tests for models.py
"""
from src.models import get_model

def test_get_model_sklearn():
    lr = get_model("logistic_regression")
    rf = get_model("random_forest")
    xgb = get_model("xgboost")
    cat = get_model("catboost")
    assert lr.__class__.__name__ == "LogisticRegression"
    assert rf.__class__.__name__ == "RandomForestClassifier"
    assert xgb.__class__.__name__ == "XGBClassifier"
    assert cat.__class__.__name__ == "CatBoostClassifier"

def test_get_model_pytorch():
    try:
        model = get_model("pytorch_mlp", input_dim=5, hidden_dim=8, dropout=0.1)
        # Check that model is a torch.nn.Module
        import torch.nn as nn
        assert isinstance(model, nn.Module)
    except ImportError:
        pass  # PyTorch not installed, skip
