import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Load your saved files
    model = joblib.load("xgb_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    custom_threshold = joblib.load("decision_threshold.pkl")
    return model, preprocessor, custom_threshold

model, preprocessor, custom_threshold = load_assets()

# --- 2. UI DESIGN ---
st.title("🛡️ Credit Default Risk Model")
st.markdown("Recall-prioritized AI decision system")

st.header("Customer Information")

# Use columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Person Age", 18, 80, 25)
    income = st.number_input("Annual Income", 0, 1000000, 50000)
    emp_len = st.number_input("Employment Length (years)", 0.0, 50.0, 2.0)
    home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])

with col2:
    loan_amnt = st.number_input("Loan Amount", 0, 100000, 10000)
    int_rate = st.number_input("Interest Rate (%)", 0.0, 30.0, 11.0)
    loan_intent = st.selectbox("Loan Intent", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])
    loan_grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])

hist_default = st.selectbox("Historical Default on File?", ["N", "Y"])
cred_hist_len = st.number_input("Credit History Length (years)", 0, 50, 5)

# --- 3. THE CALCULATION (Order Matters!) ---

if st.button("Analyze Risk"):
    # STEP 1: Create the DataFrame
    input_df = pd.DataFrame({
        'person_age': [age],
        'person_income': [income],
        'person_emp_length': [emp_len],
        'loan_amnt': [loan_amnt],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'loan_int_rate': [int_rate],
        'person_home_ownership': [home_ownership],
        'cb_person_default_on_file': [1 if hist_default == 'Y' else 0],
        'cb_person_cred_hist_length': [cred_hist_len],
        'loan_percent_income': [loan_amnt / (income + 1e-6)]
    })

    # STEP 2: Feature Engineering (Calculating the 4 extra columns)
    input_df['loan_to_income'] = input_df['loan_amnt'] / (input_df['person_income'] + 1e-6)
    input_df['loan_per_emp_year'] = input_df['loan_amnt'] / (input_df['person_emp_length'] + 1e-6)
    input_df['loan_per_age'] = input_df['loan_amnt'] / (input_df['person_age'] + 1e-6)
    input_df['is_new_worker'] = (input_df['person_emp_length'] == 0).astype(int)

    # STEP 3: Transform using your preprocessor
    X_input = preprocessor.transform(input_df)

    # STEP 4: Predict Probability (THIS CREATES 'proba')
    # We take [0] because we only have 1 row of data
    proba = model.predict_proba(X_input)[:, 1][0]

    # STEP 5: Apply Threshold (NOW 'proba' exists and can be used)
    prediction = int(proba >= custom_threshold)

    # --- 4. DISPLAY RESULTS ---
    st.divider()
    if prediction == 1:
        st.error(f"### Result: HIGH RISK (Potential Default)")
    else:
        st.success(f"### Result: LOW RISK (Safe to Lend)")
    
    st.write(f"**Default Probability:** {proba:.2%}")
    st.write(f"**Applied Threshold:** {custom_threshold:.4f}")
    st.progress(float(proba))