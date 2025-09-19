"""
main.py
Entry point for running the full pipeline.
"""

from .train import train_model
from .evaluate import evaluate_classification_model
from .utils import save_model, save_results, set_seed
import os


def main():
    set_seed(42)
    train_path = os.path.join("..", "credit-worthiness-prediction", "Train.csv")
    test_path = os.path.join("..", "credit-worthiness-prediction", "Test.csv")
    numeric_features = [
        "applicantincome", "coapplicantincome", "loanamount", "loan_amount_term", "total_income",
        "loan_repayment_rate", "loan_amount_ratio", "loan_to_income_ratio", "loan_repayment_income_ratio",
        "loan_repayment_applicatnt_income_ratio", "loan_income_thru_term", "loan_term_income_ratio"
    ]

    # Example 1: Ensemble of Logistic Regression, Random Forest, and CatBoost
    ensemble_models = ["logistic_regression", "random_forest", "catboost"]
    pipeline_ensemble = train_model(train_path, test_path, ensemble_models, numeric_features, target_col="loan_status")
    import pandas as pd
    train_df = pd.read_csv(train_path)
    X = train_df[numeric_features]
    y = train_df["loan_status"]
    results_ensemble = evaluate_classification_model(pipeline_ensemble, X, y)
    print("\nEnsemble Evaluation Results:")
    for k, v in results_ensemble.items():
        print(f"{k}:\n{v}\n")
    save_model(pipeline_ensemble, os.path.join("..", "models", "credit_risk_ensemble.joblib"))
    save_results(results_ensemble, os.path.join("..", "models", "ensemble_evaluation_results.csv"))

    # Example 2: CatBoost with Optuna and MLflow experiment tracking
    print("\nRunning CatBoost Optuna experiment (with MLflow logging)...")
    pipeline_catboost_optuna = train_model(train_path, test_path, "catboost_optuna", numeric_features, target_col="loan_status", n_trials=10)
    results_catboost_optuna = evaluate_classification_model(pipeline_catboost_optuna, X, y)
    print("\nCatBoost Optuna Evaluation Results:")
    for k, v in results_catboost_optuna.items():
        print(f"{k}:\n{v}\n")
    save_model(pipeline_catboost_optuna, os.path.join("..", "models", "credit_risk_catboost_optuna.joblib"))
    save_results(results_catboost_optuna, os.path.join("..", "models", "catboost_optuna_evaluation_results.csv"))

if __name__ == "__main__":
    main()
