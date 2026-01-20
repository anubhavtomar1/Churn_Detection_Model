import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Telecom Churn", layout="centered")

# Safety check
required_files = ["churn_model.pkl", "scaler.pkl", "model_features.pkl"]
for f in required_files:
    if not os.path.exists(f):
        st.error(f"Missing file: {f}")
        st.stop()

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("model_features.pkl")

st.title("📡 Telecom Customer Churn Prediction")

st.markdown("Predict whether a customer is likely to churn")

tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly = st.slider("Monthly Charges", 100.0, 5000.0, 800.0)
network_issues = st.slider("Network Issues (Last 3 Months)", 0, 20, 2)
complaints = st.slider("Complaints (Last 3 Months)", 0, 10, 1)
nps = st.slider("NPS Score", -100.0, 100.0, 20.0)
auto_pay_ui = st.selectbox("Auto Pay Enrolled", ["Yes", "No"])
auto_pay = 1 if auto_pay_ui == "Yes" else 0

input_dict = {col: 0 for col in features}

input_dict["tenure_months"] = tenure
input_dict["monthly_charges"] = monthly
input_dict["network_issues_3m"] = network_issues
input_dict["num_complaints_3m"] = complaints
input_dict["nps_score"] = nps
input_dict["auto_pay_enrolled"] = auto_pay

input_df = pd.DataFrame([input_dict])

prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

if st.button("Predict Churn"):

    if prediction == 1:
        st.error(f"⚠️ High Churn Risk — Probability: {probability:.2%}")

        st.subheader("📌 Recommended Retention Actions")

        if network_issues > 3:
            st.write("• Proactive network quality check and service credit")

        if monthly > 1000:
            st.write("• Personalized pricing or plan optimization")

        if auto_pay == 0:
            st.write("• Offer auto-pay enrollment incentive")

        if complaints > 2:
            st.write("• Priority customer support and faster resolution")

    else:
        st.success(f"✅ Low Churn Risk — Probability: {probability:.2%}")
        st.write("• Continue standard engagement and loyalty programs")
