import streamlit as st
import pandas as pd
import numpy as np
import pickle


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan NPL Risk Predictor",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open(
        "models/random_forest_npl_model.pkl",
        "rb"
    ) as file:

        model = pickle.load(file)

    return model


model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("💰 Loan NPL Risk Prediction System")

st.write(
    "Predict the probability that a loan may become "
    "Non-Performing and classify its risk level."
)


# ============================================================
# INPUTS
# ============================================================

st.subheader("Loan Information")

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=100000.0
)

total_interest_accrued = st.number_input(
    "Total Interest Accrued",
    min_value=0.0,
    value=0.0
)

total_interest_charged = st.number_input(
    "Total Interest Charged",
    min_value=0.0,
    value=0.0
)

management_fee = st.number_input(
    "Management Fee",
    min_value=0.0,
    value=0.0
)

tax = st.number_input(
    "Tax",
    min_value=0.0,
    value=0.0
)

loan_type = st.selectbox(
    "Loan Type",
    [
        "bnpl",
        "Omnipay MFC",
        "bullet",
        "cg",
        "multiproNew",
        "newbnpl",
        "paylaterPlus"
    ]
)

state = st.selectbox(
    "State",
    [
        "Lagos",
        "Rivers",
        "Abuja",
        "Oyo",
        "Ogun",
        "Kwara",
        "Bauchi",
        "Kano",
        "Other"
    ]
)


# ============================================================
# CALCULATE DERIVED FEATURES
# ============================================================

interest_rate = (
    total_interest_charged / loan_amount
    if loan_amount > 0
    else 0
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict NPL Risk"):

    input_data = pd.DataFrame({

        "LENDERNAME": ["OmniPay Limited"],
        "ORDERSOURCE": ["Distributor App"],
        "Organisation_Name": ["Unknown"],
        "State Name": [state],
        "Date": [pd.Timestamp.today()],
        "LOANDATE": [pd.Timestamp.today()],
        "LOANAMOUNT": [loan_amount],
        "Total Interest Accrued": [
            total_interest_accrued
        ],
        "Total Interest Charged": [
            total_interest_charged
        ],
        "Management Fee": [management_fee],
        "Tax": [tax],
        "LOANTYPE": [loan_type],
        "Agent Name": ["Unknown"],
        "Agent Status": ["Unknown"],
        "TeamLead": ["Unknown"],
        "ACM": ["Unknown"],
        "RCM": ["Unknown"],
        "Market Name": ["Unknown"]
    })


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[:, 1][0]


    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if probability < 0.30:

        risk_level = "Low"

    elif probability < 0.70:

        risk_level = "Medium"

    else:

        risk_level = "High"


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Predicted NPL",
            int(prediction)
        )

    with col2:

        st.metric(
            "NPL Probability",
            f"{probability:.2%}"
        )

    with col3:

        st.metric(
            "Risk Level",
            risk_level
        )


    if risk_level == "High":

        st.error(
            "⚠️ High-risk loan. Further credit review is recommended."
        )

    elif risk_level == "Medium":

        st.warning(
            "⚠️ Medium-risk loan. Additional monitoring is recommended."
        )

    else:

        st.success(
            "✅ Low-risk loan."
        )