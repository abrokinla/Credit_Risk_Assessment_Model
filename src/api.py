from data_prep import feature_eng, preprocess_dependents

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import requests
import os
import numpy as np
import pandas as pd
import lime.lime_tabular
from typing import Any, Dict

app = FastAPI()

class CreditRequest(BaseModel):
    gender: str
    married: str
    dependents: str
    education: str
    self_employed: str
    applicantincome: float
    coapplicantincome: float
    loanamount: float
    loan_amount_term: int
    credit_history: float
    property_area: str
    model_file: str  # New field to specify which model to use
    
# For LIME, you need the training data used for the model (features only)
try:
    X_train = joblib.load("models/X_train.joblib")
except Exception:
    X_train = np.random.rand(100, 5)  # fallback dummy data
    # Replace with actual X_train loading in production

feature_names = ["age", "income", "loan_amount", "loan_term", "credit_score"]

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=["Not Creditworthy", "Creditworthy"],
    discretize_continuous=True,
    mode="classification"
)

@app.get("/")
def read_root():
    return {"message": "Credit Risk Assessment API is running"}

@app.post("/predict")
def predict_credit(request: CreditRequest):
    # Dynamically load the selected model
    model_path = os.path.join("..", "models", request.model_file)
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading error: {e}")

    # Define feature columns used in training
    numeric_features = [
        "applicant_income", "coapplicant_income", "loan_amount", "loan_amount_term", "total_income",
        "loan_repayment_rate", "loan_amount_ratio", "loan_to_income_ratio", "loan_repayment_income_ratio",
        "loan_repayment_applicant_income_ratio", "loan_income_thru_term", "loan_term_income_ratio",
        "dependents_0", "dependents_1", "dependents_2", "dependents_3"
    ]

    # Prepare input data for prediction
    input_data = {
        "gender": request.gender,
        "married": request.married,
        "dependents": request.dependents,
        "education": request.education,
        "self_employed": request.self_employed,
        "applicantincome": request.applicantincome,
        "coapplicantincome": request.coapplicantincome,
        "loanamount": request.loanamount,
        "loan_amount_term": request.loan_amount_term,
        "credit_history": request.credit_history,
        "property_area": request.property_area
    }
    input_df = pd.DataFrame([input_data])

    # Rename columns to match training data format
    column_mapping = {
        'applicantincome': 'applicant_income',
        'coapplicantincome': 'coapplicant_income',
        'loanamount': 'loan_amount'
    }
    input_df = input_df.rename(columns=column_mapping)

    # Scale amounts to match training data units (divide by 1000)
    input_df['applicant_income'] /= 1000
    input_df['coapplicant_income'] /= 1000
    input_df['loan_amount'] /= 1000

    input_df = preprocess_dependents(input_df)
    input_df = feature_eng(input_df)
    X = input_df[numeric_features]

    # Get prediction and probability
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    # Generate LIME explanation
    try:
        # Use processed features (16 features) for LIME explanation
        # Create proxy training data for LIME (not actual training set)
        proxy_training_data = np.random.rand(100, 16)  # Dummy data for statistics

        # Create explainer for processed features
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=proxy_training_data,
            feature_names=numeric_features,  # 16 processed feature names
            class_names=["Not Approved", "Approved"],
            mode="classification"
        )

        # Generate explanation using the processed instance
        exp = explainer.explain_instance(
            X.iloc[0].values.astype(float),
            model.predict_proba,
            num_features=5
        )
        # Get the top factors influencing the prediction
        explanation_list = exp.as_list()
        explanation_text = "\nTop factors influencing this decision:\n"
        for feature, impact in explanation_list:
            impact = round(float(impact), 3)  # Ensure impact is converted to float
            feature_str = str(feature).replace(" <= ", " less than or equal to ")
            feature_str = feature_str.replace(" > ", " greater than ")
            if impact > 0:
                explanation_text += f"- {feature_str} increases approval chance (impact: +{impact})\n"
            else:
                explanation_text += f"- {feature_str} decreases approval chance (impact: {impact})\n"
    except Exception as e:
        explanation_text = f"Could not generate detailed explanation: {str(e)}"

    # Create human-readable prediction
    approval_status = "Approved" if prediction == 1 else "Not Approved"
    confidence = round(max(probability) * 100, 2)
    prediction_text = f"Loan Application Status: {approval_status} (Confidence: {confidence}%)"

    return {
        "prediction": prediction_text,
        "lime_explanation": explanation_text,
        "raw_prediction": int(prediction),
        "confidence": confidence
    }
