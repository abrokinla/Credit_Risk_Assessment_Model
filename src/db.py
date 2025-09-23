from sqlalchemy import create_engine, text
import pandas as pd
import os

engine = create_engine("postgresql+psycopg2://credit_user:1234@localhost/credit_risk_db")
with engine.connect() as conn:
    result = conn.execute(text("SELECT 1;"))
    print(result.fetchone())

def insert_dataframe_to_db(df, table_name="credit_requests"):
    # Use SQLAlchemy's to_sql for bulk insert
    df.to_sql(table_name, engine, if_exists='append', index=False)

if __name__ == "__main__":
    # Load Train.csv
    train_df = pd.read_csv(os.path.abspath("../Credit_Risk_Assessment_Model/credit-worthiness-prediction/Train.csv"))
    # Rename columns to match DB schema if needed
    train_df = train_df.rename(columns={
        "ApplicantIncome": "applicant_income",
        "CoapplicantIncome": "coapplicant_income",
        "LoanAmount": "loan_amount",
        "Loan_Amount_Term": "loan_amount_term",
        "Credit_History": "credit_history",
        "Property_Area": "property_area",
        "Loan_Status": "loan_status",
        "Total_Income": "total_income",
        "Loan_ID": "loan_id",
        "Gender": "gender",
        "Married": "married",
        "Dependents": "dependents",
        "Education": "education",
        "Self_Employed": "self_employed",
        "ID": "id"
    })
    # Insert into DB
    insert_dataframe_to_db(train_df)

def load_data_from_db(table_name="credit_requests", where_clause=None):
    """
    Load data from the database table as a pandas DataFrame.
    Optionally filter rows using a WHERE clause.
    """
    query = f"SELECT * FROM {table_name}"
    if where_clause:
        query += f" WHERE {where_clause}"
    return pd.read_sql(query, engine)