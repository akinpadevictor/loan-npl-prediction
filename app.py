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
# INPUT SECTION
# ============================================================

st.subheader("Loan Information")


col1, col2 = st.columns(2)


with col1:

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=100000.0,
        step=10000.0
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


with col2:

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

    lender_name = st.text_input(
        "Lender Name",
        value="OmniPay Limited"
    )

    order_source = st.text_input(
        "Order Source",
        value="Distributor App"
    )

    organisation_name = st.text_input(
        "Organisation Name",
        value="Unknown"
    )


# ============================================================
# DERIVED FEATURES
# ============================================================

loan_date = pd.Timestamp.today().normalize()


# Interest Rate
interest_rate = (
    total_interest_charged / loan_amount
    if loan_amount > 0
    else 0
)


# Loan date features
loan_month = loan_date.month
loan_year = loan_date.year
loan_day_of_week = loan_date.dayofweek


# Loan Amount Band
if loan_amount <= 1_000_000:

    loan_amount_band = "<1M"

elif loan_amount <= 5_000_000:

    loan_amount_band = "1M-5M"

elif loan_amount <= 10_000_000:

    loan_amount_band = "5M-10M"

elif loan_amount <= 50_000_000:

    loan_amount_band = "10M-50M"

else:

    loan_amount_band = "50M+"


# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict NPL Risk", type="primary"):

    # --------------------------------------------------------
    # CREATE EXACT 21 MODEL FEATURES
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "LOANAMOUNT": [loan_amount],

        "Total Interest Accrued": [
            total_interest_accrued
        ],

        "Total Interest Charged": [
            total_interest_charged
        ],

        "Management Fee": [
            management_fee
        ],

        "Tax": [
            tax
        ],

        "Interest_Rate": [
            interest_rate
        ],

        "Loan_Month": [
            loan_month
        ],

        "Loan_Year": [
            loan_year
        ],

        "Loan_DayOfWeek": [
            loan_day_of_week
        ],

        "LENDERNAME": [
            lender_name
        ],

        "ORDERSOURCE": [
            order_source
        ],

        "Organisation_Name": [
            organisation_name
        ],

        "State Name": [
            state
        ],

        "LOANTYPE": [
            loan_type
        ],

        "Agent Name": [
            "Unknown"
        ],

        "Agent Status": [
            "Unknown"
        ],

        "TeamLead": [
            "Unknown"
        ],

        "ACM": [
            "Unknown"
        ],

        "RCM": [
            "Unknown"
        ],

        "Market Name": [
            "Unknown"
        ],

        "Loan_Amount_Band": [
            loan_amount_band
        ]
    })


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[:, 1][0]


    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------
    #
    # Consistent with the final CSV:
    #
    # < 0.40       = Low Risk
    # 0.40 - 0.69  = Medium Risk
    # >= 0.70      = High Risk
    #

    if probability >= 0.70:

        risk_level = "High Risk"

    elif probability >= 0.40:

        risk_level = "Medium Risk"

    else:

        risk_level = "Low Risk"


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


    # --------------------------------------------------------
    # RISK MESSAGE
    # --------------------------------------------------------

    if risk_level == "High Risk":

        st.error(
            "⚠️ High-risk loan. Further credit review is recommended."
        )

    elif risk_level == "Medium Risk":

        st.warning(
            "⚠️ Medium-risk loan. Additional monitoring is recommended."
        )

    else:

        st.success(
            "✅ Low-risk loan."
        )


    # --------------------------------------------------------
    # SHOW MODEL INPUTS
    # --------------------------------------------------------

    with st.expander("View model inputs"):

        st.dataframe(
            input_data,
            use_container_width=True
        )

