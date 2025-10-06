import sys
import streamlit as st
import requests
import os
import pandas as pd

sys.path.append(os.path.abspath(".."))
from src.data_prep import feature_eng
st.title("Credit Risk Assessment App")
st.write("Enter applicant details to assess creditworthiness.")

# Model selection
model_dir = os.path.abspath("../models")
model_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
if not model_files:
    st.warning("No models found in the models directory. Please train and save your models first.")
selected_model_file = st.selectbox("Select Model", model_files if model_files else ["No models available"])

# User input fields matching database columns (except ID, Loan_ID)
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0, value=50000)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
loan_amount_term = st.number_input("Loan Amount Term (months)", min_value=1, value=360)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict Creditworthiness", key="predict_btn") and model_files:
    data = {
    "gender": gender,
    "married": married,
    "dependents": dependents,
    "education": education,
    "self_employed": self_employed,
    "applicantincome": applicant_income,
    "coapplicantincome": coapplicant_income,
    "loanamount": loan_amount,
    "loan_amount_term": loan_amount_term,
    "credit_history": credit_history,
    "property_area": property_area,
    "model_file": selected_model_file
    }

    # Apply feature engineering before sending to API (optional, for local prediction)
    input_df = pd.DataFrame([data])
    input_df = feature_eng(input_df)
    # If you want to send engineered features, update data dict
    data.update(input_df.iloc[0].to_dict())
    try:
        response = requests.post("http://localhost:8000/predict", json=data)
        if response.status_code == 200:
            result = response.json()
            
            # Display prediction with confidence
            st.success(result['prediction'])
            
            # Display explanation in an expander
            with st.expander("View Detailed Explanation"):
                st.markdown(result['lime_explanation'])
            
            # Display confidence separately if needed
            st.info(f"Prediction Confidence: {result['confidence']}%")
            
            # Optional: Display visualization or additional details
            if result['raw_prediction'] == 1:
                st.balloons()  # Celebrate approved loans
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")
elif st.button("Predict Creditworthiness", key="predict_btn_disabled") and not model_files:
    st.error("No models available for prediction. Please train and save your models.")