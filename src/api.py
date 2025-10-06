from data_prep import feature_eng

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
    input_df = feature_eng(input_df)

    # Get prediction and probability
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    # Generate LIME explanation
    try:
        # Use only original features for LIME explanation
        original_features = [
            "gender", "married", "dependents", "education", "self_employed",
            "applicantincome", "coapplicantincome", "loanamount", "loan_amount_term",
            "credit_history", "property_area"
        ]
        
        # Prepare categorical features for LIME
        categorical_features = ['gender', 'married', 'dependents', 'education', 'self_employed', 'property_area']
        categorical_names = {}
        
        # Create a copy with only original features
        lime_df = pd.DataFrame([input_data])  # Use original data, not engineered features
        
        # Convert categorical variables to numeric
        for feature in categorical_features:
            if feature in lime_df.columns:
                # Create a mapping of categories to numbers
                unique_values = pd.Categorical(lime_df[feature]).categories
                categorical_names[lime_df.columns.get_loc(feature)] = unique_values
                lime_df[feature] = pd.Categorical(lime_df[feature]).codes
        
        # Create explainer with categorical features
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=lime_df.values,
            feature_names=original_features,
            class_names=["Not Approved", "Approved"],
            categorical_features=[lime_df.columns.get_loc(col) for col in categorical_features if col in lime_df.columns],
            categorical_names=categorical_names,
            mode="classification"
        )
        
        # Generate explanation
        exp = explainer.explain_instance(
            lime_df.iloc[0].values,
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