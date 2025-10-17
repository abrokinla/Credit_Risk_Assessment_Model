# Credit Risk Assessment Model

This project provides a robust, modular machine learning pipeline for credit risk assessment using tabular data. It is designed for clarity, reproducibility, and ease of deployment. The pipeline supports advanced model ensembling, hyperparameter optimization with experiment tracking, and is fully testable for production use.

## Project Structure

```
Credit_Risk_Assessment_Model/
│
├── credit-worthiness-prediction/   # Data files (Train.csv, Test.csv, etc.)
├── models/                         # Saved models and evaluation results
├── src/                            # Source code modules
│   ├── data_prep.py                # Data loading, cleaning, feature engineering
│   ├── pipeline.py                 # Preprocessing and modeling pipelines
│   ├── models.py                   # Model definitions (sklearn, CatBoost, PyTorch, ensemble)
│   ├── train.py                    # Training logic, ensemble, Optuna + MLflow
│   ├── evaluate.py                 # Model evaluation and metrics
│   ├── utils.py                    # Utility functions (save/load, seed, etc.)
│   └── main.py                     # Pipeline entry point
├── tests/                          # Unit tests for all modules
│   └── ...                         # test_data_prep.py, test_pipeline.py, etc.
└── README.md                       # Project documentation
```


## Features
- **Modular codebase**: Each stage (data preparation, modeling, training, evaluation) is separated for maintainability and clarity.
- **Model ensembling**: Combine predictions from multiple models (e.g., Logistic Regression, Random Forest, CatBoost) for improved accuracy and robustness.
- **Hyperparameter optimization & experiment tracking**: Automated CatBoost tuning with Optuna, with all experiments and results logged via MLflow for full reproducibility.
- **Reproducibility**: Deterministic results via random seed setting, requirements export, and comprehensive unit tests.
- **Easy deployment**: Trained models are exported with joblib and ready for integration into APIs or batch scoring workflows.

## Setup

1. **Clone the repository**
2. **Create and activate a virtual environment**
   ```bash
   python -m venv env
   # On Windows:
   env\Scripts\activate
   # On macOS/Linux:
   source env/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install numpy pandas scikit-learn xgboost catboost optuna mlflow joblib pytest
   # Add torch torchvision if using PyTorch models
   ```
4. **Export requirements**
   ```bash
   pip freeze > requirements.txt
   ```

## Usage

1. **Prepare your data**
   - Place `Train.csv` and `Test.csv` in the `credit-worthiness-prediction/` folder.
2. **Set up the database**
   ```bash
   python src/db.py
   ```
   - This creates the database table and populates it with the CSV data.
3. **Run the pipeline**
   ```bash
   python -m src.main
   ```
   - This will train and evaluate both an ensemble and a CatBoost model with Optuna tuning.
   - Models and evaluation results are saved in the `models/` folder.
4. **Run tests**
   ```bash
   pytest tests/
   ```

## Experiment Tracking
- CatBoost hyperparameter optimization is tracked with MLflow.
- To view experiments, run:
  ```bash
  mlflow ui
  ```
  and open the provided URL in your browser.

## Customization
- To use different models or features, edit `src/main.py` and `src/train.py`.
- To add new data sources (e.g., a database), update `src/data_prep.py`.

## Deployment

### Streamlit Cloud (Recommended for Personal Projects)
1. **Push this repo to GitHub** (public repository).
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Deploy from GitHub**:
   - Connect your repository
   - Select the `main` branch
   - Set `app.py` as the main file
   - The app will automatically detect `requirements.txt` and install dependencies
4. **Get your public URL** - instantly accessible for showcasing your work!

### Local Streamlit App
```bash
streamlit run app.py
```

### Alternative: Heroku/Free Hosting
For more control or if you need Flask/FastAPI:
```bash
git push heroku main
```

## Requirements
See `requirements.txt` for all dependencies.

## License
MIT License
