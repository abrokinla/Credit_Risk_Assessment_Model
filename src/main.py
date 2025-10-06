def main():
    import os
    import numpy as np
    from sklearn.model_selection import train_test_split
    from utils import set_seed, save_model, save_results
    from evaluate import evaluate_classification_model
    from data_prep import load_data, combine_and_preprocess
    from train import train_model

    set_seed(42)

    # Load and preprocess data
    train_df, test_df = load_data(from_db=True)
    combined_df = combine_and_preprocess(train_df, test_df)
    train_df_preprocessed = combined_df[combined_df['source'] == 'train']
    test_df_preprocessed = combined_df[combined_df['source'] == 'test']

    numeric_features = [
        "applicant_income", "coapplicant_income", "loan_amount", "loan_amount_term", "total_income",
        "loan_repayment_rate", "loan_amount_ratio", "loan_to_income_ratio", "loan_repayment_income_ratio",
        "loan_repayment_applicant_income_ratio", "loan_income_thru_term", "loan_term_income_ratio"
    ]
    target_col = "loan_status"

    X = train_df_preprocessed[numeric_features]
    y = np.where(train_df_preprocessed[target_col] == 1, 1, 0)

    # Run CatBoost Optuna 
    print("\nRunning CatBoost Optuna experiment (with MLflow logging)...")
    pipeline_catboost_optuna = train_model(
        X=X,
        test_df=test_df_preprocessed,
        model_name="catboost_optuna",
        numeric_features=numeric_features,
        y=y,
        n_trials=10
    )

    # fit model before evaluation
    pipeline_catboost_optuna.fit(X, y)

    results_catboost_optuna = evaluate_classification_model(pipeline_catboost_optuna, X, y)
    print("\nCatBoost Optuna Evaluation Results:")
    for k, v in results_catboost_optuna.items():
        print(f"{k}:\n{v}\n")

    save_model(
        pipeline_catboost_optuna,
        os.path.join("..", "Credit_Risk_Assessment_Model/models", "credit_risk_catboost_optuna.joblib")
    )
    save_results(
        results_catboost_optuna,
        os.path.join("..", "Credit_Risk_Assessment_Model/models", "catboost_optuna_evaluation_results.csv")
    )

    # Ensemble with optimized CatBoost
    print("\nBuilding Ensemble with Optimized CatBoost...")

    ensemble_models = ["logistic_regression", "random_forest", "catboost_optuna"]
    pipeline_ensemble = train_model(
        X=X,
        test_df=test_df_preprocessed,
        model_name=ensemble_models,
        numeric_features=numeric_features,
        y=y,
        n_trials=10 
    )

    pipeline_ensemble.fit(X, y)

    results_ensemble = evaluate_classification_model(pipeline_ensemble, X, y)
    print("\nEnsemble Evaluation Results:")
    for k, v in results_ensemble.items():
        print(f"{k}:\n{v}\n")

    save_model(
        pipeline_ensemble,
        os.path.join("..", "Credit_Risk_Assessment_Model/models", "credit_risk_ensemble.joblib")
    )
    save_results(
        results_ensemble,
        os.path.join("..", "Credit_Risk_Assessment_Model/models", "ensemble_evaluation_results.csv")
    )


if __name__ == "__main__":
    main()
