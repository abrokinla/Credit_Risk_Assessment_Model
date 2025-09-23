
import streamlit as st
import requests

st.title("Credit Risk Assessment App")
st.write("Enter applicant details to assess creditworthiness.")

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

if st.button("Predict Creditworthiness"):
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
        "property_area": property_area
    }
    try:
        response = requests.post("http://localhost:8000/predict", json=data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result['prediction']}")
            st.info(f"Reason: {result['lime_explanation']}")
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")
