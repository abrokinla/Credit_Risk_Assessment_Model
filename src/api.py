
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import lime.lime_tabular
from typing import Any, Dict

app = FastAPI()


# Update CreditRequest to include all fields from the frontend, including 'id'
class CreditRequest(BaseModel):
    id: str
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

# Load model and training data for LIME
model = joblib.load("models/credit_risk_model.joblib")
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
def predict_credit(request: CreditRequest) -> Dict[str, Any]:
    # Convert input to DataFrame
    # Prepare input for model (drop id, ensure correct order)
    input_dict = request.dict()
    user_id = input_dict.pop("id")
    input_data = pd.DataFrame([input_dict])
    # Predict
    try:
        proba = model.predict_proba(input_data)[0, 1]
        pred = int(proba >= 0.5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # LIME explanation
    try:
        exp = explainer.explain_instance(input_data.values[0], model.predict_proba, num_features=5)
        explanation = exp.as_list()
        explanation_str = "; ".join([f"{feat}: {weight:+.3f}" for feat, weight in explanation])
    except Exception as e:
        explanation_str = f"LIME explanation error: {e}"

    return {
        "id": user_id,
        "prediction": "Creditworthy" if pred == 1 else "Not Creditworthy",
        "probability": proba,
        "lime_explanation": explanation_str
    }