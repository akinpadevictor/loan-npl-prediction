import streamlit as st
import pandas as pd
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
# LOAD TRAINED MODEL
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
# PAGE TITLE
# ============================================================

st.title("💰 Loan NPL Risk Predictor")

st.write(
    "Enter loan information below to estimate the probability "
    "that the loan may become a Non-Performing Loan (NPL)."
)


# ============================================================
# INPUT SECTION
# ============================================================

st.header("Loan Information")


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Numeric Inputs
# ------------------------------------------------------------

with col1:

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=1.0,
        value=100000.0,
        step=1000.0
    )

    total_interest_accrued = st.number_input(
        "Total Interest Accrued",
        min_value=0.0,
        value=0.0,
        step=1000.0
    )

    total_interest_charged = st.number_input(
        "Total Interest Charged",
        min_value=0.0,
        value=0.0,
        step=1000.0
    )

    management_fee = st.number_input(
        "Management Fee",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    tax = st.number_input(
        "Tax",
        min_value=0.0,
        value=0.0,
        step=100.0
    )


# ------------------------------------------------------------
# Categorical Inputs
# ------------------------------------------------------------

with col2:

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


# ============================================================
# STATE
# ============================================================

# These are based on the State Name categories found
# in the training data.

state_name = st.selectbox(
    "State",
    [
        "Abuja Fed. Capital",
        "Adamawa",
        "Akwa Ibom",
        "Anambra",
        "Bauchi",
        "Bayelsa",
        "Benue",
        "Borno",
        "Cross River",
        "Delta",
        "Ebonyi",
        "Edo",
        "Ekiti",
        "Enugu",
        "Gombe",
        "Imo",
        "Jigawa",
        "Kaduna",
        "Kano",
        "Katsina",
        "Kebbi",
        "Kogi",
        "Kwara",
        "Lagos",
        "Nasarawa",
        "Niger",
        "Ogun",
        "Ondo",
        "Osun",
        "Oyo",
        "Plateau",
        "Rivers"
    ]
)


# ============================================================
# OPTIONAL AGENT / MANAGEMENT INFORMATION
# ============================================================

st.subheader("Agent / Management Information")

col3, col4, col5 = st.columns(3)

with col3:

    agent_name = st.text_input(
        "Agent Name",
        value="Unknown"
    )

    agent_status = st.text_input(
        "Agent Status",
        value="Unknown"
    )


with col4:

    team_lead = st.text_input(
        "Team Lead",
        value="Unknown"
    )

    acm = st.text_input(
        "ACM",
        value="Unknown"
    )


with col5:

    rcm = st.text_input(
        "RCM",
        value="Unknown"
    )

    market_name = st.text_input(
        "Market Name",
        value="Unknown"
    )


# ============================================================
# DERIVED FEATURES
# ============================================================

loan_date = pd.Timestamp.today().normalize()


# Interest Rate
if loan_amount > 0:

    interest_rate = (
        total_interest_charged / loan_amount
    )

else:

    interest_rate = 0.0


# Date-derived features

loan_month = loan_date.month

loan_year = loan_date.year

loan_day_of_week = loan_date.dayofweek


# ============================================================
# LOAN AMOUNT BAND
# ============================================================

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
# CREATE MODEL INPUT
# ============================================================

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
        state_name
    ],

    "LOANTYPE": [
        loan_type
    ],

    "Agent Name": [
        agent_name
    ],

    "Agent Status": [
        agent_status
    ],

    "TeamLead": [
        team_lead
    ],

    "ACM": [
        acm
    ],

    "RCM": [
        rcm
    ],

    "Market Name": [
        market_name
    ],

    "Loan_Amount_Band": [
        loan_amount_band
    ]
})


# ============================================================
# ENSURE EXACT MODEL FEATURE ORDER
# ============================================================

model_features = [

    "LOANAMOUNT",
    "Total Interest Accrued",
    "Total Interest Charged",
    "Management Fee",
    "Tax",
    "Interest_Rate",
    "Loan_Month",
    "Loan_Year",
    "Loan_DayOfWeek",
    "LENDERNAME",
    "ORDERSOURCE",
    "Organisation_Name",
    "State Name",
    "LOANTYPE",
    "Agent Name",
    "Agent Status",
    "TeamLead",
    "ACM",
    "RCM",
    "Market Name",
    "Loan_Amount_Band"
]


input_data = input_data[model_features]


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔍 Predict NPL Risk",
    use_container_width=True
):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[:, 1][0]


    # ========================================================
    # RISK CLASSIFICATION
    # ========================================================

    if probability < 0.30:

        risk_level = "Low"

    elif probability < 0.70:

        risk_level = "Medium"

    else:

        risk_level = "High"


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    st.header("Prediction Result")


    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Predicted NPL",
            "Yes" if prediction == 1 else "No"
        )


    with result_col2:

        st.metric(
            "NPL Probability",
            f"{probability:.2%}"
        )


    with result_col3:

        st.metric(
            "Risk Level",
            risk_level
        )


    # ========================================================
    # RISK MESSAGE
    # ========================================================

    if risk_level == "High":

        st.error(
            "⚠️ High Risk: This loan has a high estimated "
            "probability of becoming an NPL."
        )

    elif risk_level == "Medium":

        st.warning(
            "⚠️ Medium Risk: This loan requires additional "
            "monitoring and assessment."
        )

    else:

        st.success(
            "✅ Low Risk: This loan has a relatively low "
            "estimated probability of becoming an NPL."
        )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.subheader("NPL Probability")

    st.progress(
        float(probability)
    )


    # ========================================================
    # MODEL INPUTS
    # ========================================================

    with st.expander(
        "View Model Input Data"
    ):

        st.dataframe(
            input_data,
            use_container_width=True
        )
