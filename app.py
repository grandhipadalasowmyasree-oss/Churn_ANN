import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib

from tensorflow.keras.models import load_model


# Load model
model = load_model("bank_churn_ann.h5")

# Load scaler
scaler = joblib.load("scaler.pkl")

# Load feature names
with open("feature_names.json", "r") as f:
    feature_names = json.load(f)


# Page configuration
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦"
)

st.title("🏦 Bank Customer Churn Prediction")

st.write(
    "Enter customer details to predict whether "
    "the customer is likely to churn or stay."
)


# Customer inputs

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

num_products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1
)

has_card = st.selectbox(
    "Has Credit Card?",
    ["Yes", "No"]
)

active_member = st.selectbox(
    "Is Active Member?",
    ["Yes", "No"]
)

salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)


# Prediction
if st.button("Predict Churn"):

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=feature_names
    )

    input_data["CreditScore"] = credit_score
    input_data["Age"] = age
    input_data["Tenure"] = tenure
    input_data["Balance"] = balance
    input_data["NumOfProducts"] = num_products

    input_data["HasCrCard"] = (
        1 if has_card == "Yes" else 0
    )

    input_data["IsActiveMember"] = (
        1 if active_member == "Yes" else 0
    )

    input_data["EstimatedSalary"] = salary


    # Geography encoding

    if "Geography_Germany" in input_data.columns:
        input_data["Geography_Germany"] = (
            1 if geography == "Germany" else 0
        )

    if "Geography_Spain" in input_data.columns:
        input_data["Geography_Spain"] = (
            1 if geography == "Spain" else 0
        )


    # Gender encoding

    if "Gender_Male" in input_data.columns:
        input_data["Gender_Male"] = (
            1 if gender == "Male" else 0
        )


    # Scaling

    input_scaled = scaler.transform(input_data)


    # Prediction

    probability = model.predict(
        input_scaled,
        verbose=0
    )[0][0]


    # Result

    if probability >= 0.5:

        st.error(
            "⚠️ Customer is likely to CHURN"
        )

    else:

        st.success(
            "✅ Customer is likely to STAY"
        )


    st.write(
        "Churn Probability:",
        round(float(probability) * 100, 2),
        "%"
    )