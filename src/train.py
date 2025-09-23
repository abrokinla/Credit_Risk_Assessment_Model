"""
train.py
Handles model training and hyperparameter tuning.
"""

from .data_prep import load_data, combine_and_preprocess
from .models import get_model, EnsembleAverager
from .pipeline import get_pipeline
import pandas as pd

def train_model(X=None, test_df=None, model_name=None, numeric_features=None, y=None, target_col="loan_status", **model_kwargs):
    """
    Trains a model and returns the fitted pipeline.
    If X and y are provided, uses them directly for training.
    If not, loads and preprocesses data from file paths.
    Supports ensemble training if model_name is a list.
    If 'catboost_optuna' is in the list, uses the optimized CatBoost model in the ensemble.
    """
    if X is not None and y is not None:
        # Use provided X and y directly
        pass
    else:
        train_df, test_df = load_data(train_path, test_path)
        combined_df = combine_and_preprocess(train_df, test_df)
        train_df_preprocessed = combined_df[combined_df['source'] == 'train']
        X = train_df_preprocessed[numeric_features]
        y = train_df_preprocessed[target_col]

    # --- ENSEMBLE LOGIC WITH OPTIMIZED CATBOOST ---
    if isinstance(model_name, list):
        pipelines = []
        for name in model_name:
            if name == "catboost_optuna":
                # Train and use the optimized CatBoost model
                pipe = train_catboost_optuna(X, y, numeric_features, **model_kwargs)
            else:
                safe_kwargs = {k: v for k, v in model_kwargs.items() if k not in ["n_trials", "random_state"]}
                model = get_model(name, **model_kwargs)
                pipe = get_pipeline(model, numeric_features)
                pipe.fit(X, y)
            pipelines.append(pipe)
        ensemble = EnsembleAverager([p.named_steps['model'] for p in pipelines])
        # Fit the ensemble
        ensemble.fit(X, y)
        # Wrap in a pipeline for compatibility
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("preprocessor", pipelines[0].named_steps['preprocessor']),
            ("model", ensemble)
        ])
    elif model_name == "catboost_optuna":
        return train_catboost_optuna(X, y, numeric_features, **model_kwargs)
    else:
        model = get_model(model_name, **model_kwargs)
        pipeline = get_pipeline(model, numeric_features)
        pipeline.fit(X, y)
        return pipeline


# Optuna experiment tracking for CatBoost
def train_catboost_optuna(X, y, numeric_features, n_trials=20, random_state=42):
    import optuna
    from catboost import CatBoostClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import mlflow
    import mlflow.catboost

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 300, 1000),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
            'max_bin': trial.suggest_int('max_bin', 100, 400),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
            'auto_class_weights': 'SqrtBalanced',
            'eval_metric': 'AUC',
            'random_state': random_state,
            'verbose': False,
            'allow_writing_files': False
        }

        if params["bootstrap_type"] == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif params["bootstrap_type"] == "Bernoulli":
            params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)

        return auc

    mlflow.set_experiment("catboost_optuna")
    with mlflow.start_run(run_name="catboost_optuna_experiment"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_value = study.best_value

        # Log only best results
        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_auc", best_value)

        # Train best model on full data
        best_model = CatBoostClassifier(
            **best_params,
            auto_class_weights='SqrtBalanced',
            eval_metric='AUC',
            random_state=random_state,
            verbose=False,
            allow_writing_files=False
        )
        best_model.fit(X, y)

        mlflow.catboost.log_model(best_model, "catboost_model")

    # Return as a pipeline for compatibility
    from .pipeline import get_pipeline
    return get_pipeline(best_model, numeric_features)
