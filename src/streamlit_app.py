import streamlit as st
import os
import pandas as pd
import joblib
import numpy as np
import lime.lime_tabular
from .data_prep import preprocess_dependents, feature_eng

def run():
    st.title("Credit Risk Assessment App")
    st.write("Enter applicant details to assess creditworthiness.")

    # Model selection
    model_dir = "models"
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")] if os.path.exists(model_dir) else []
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
        # Define feature columns used in training
        numeric_features = [
            "applicant_income", "coapplicant_income", "loan_amount", "loan_amount_term", "total_income",
            "loan_repayment_rate", "loan_amount_ratio", "loan_to_income_ratio", "loan_repayment_income_ratio",
            "loan_repayment_applicant_income_ratio", "loan_income_thru_term", "loan_term_income_ratio",
            "dependents_0", "dependents_1", "dependents_2", "dependents_3"
        ]

        # Prepare input data for prediction
        input_data = {
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

        # Load model
        model_path = os.path.join(model_dir, selected_model_file)
        model = joblib.load(model_path)

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

        # Display prediction with confidence
        st.success(prediction_text)

        # Display explanation in an expander
        with st.expander("View Detailed Explanation"):
            st.markdown(explanation_text)

        # Optional: Celebrate approved loans
        if prediction == 1:
            st.balloons()

    elif st.button("Predict Creditworthiness", key="predict_btn_disabled") and not model_files:
        st.error("No models available for prediction. Please train and save your models.")
