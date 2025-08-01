import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64


@st.cache_data
def load_data():
    return pd.read_csv('data/customer_churn.csv')

@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

df = load_data()
model = load_model()

st.title("📊 Telco Customer Churn Prediction")

st.subheader("🔍 Explore Dataset")

with st.expander("Preview Dataset"):
    st.write(df.head())

def get_csv_download_link(dataframe):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="telco_churn.csv">📥 Download Dataset as CSV</a>'
    return href

st.markdown(get_csv_download_link(df), unsafe_allow_html=True)

st.subheader("🧾 Enter Customer Details")

def user_input():
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=2000.0)

    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    return pd.DataFrame([data])

input_df = user_input()

st.subheader("🔮 Churn Prediction")

if st.button("Predict"):
    input_processed = pd.get_dummies(input_df)
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in input_processed.columns:
            input_processed[col] = 0
    input_processed = input_processed[model_features]

    prediction = model.predict(input_processed)[0]
    prob = model.predict_proba(input_processed)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to churn (Probability: {prob:.2f})")
    else:
        st.success(f"✅ This customer is not likely to churn (Probability: {1 - prob:.2f})")

