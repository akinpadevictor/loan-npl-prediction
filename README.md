# Loan NPL Prediction & Risk Analytics

## 📌 Project Overview

This project develops a machine learning solution for predicting Non-Performing Loans (NPL) and classifying loan records into different risk levels.

The project combines Machine Learning, Python, Power BI, and data analytics to transform loan data into actionable risk insights.

The machine learning model predicts whether a loan is likely to become a Non-Performing Loan and generates an NPL probability and corresponding risk level.

## 🎯 Project Objectives

- Predict the likelihood of a loan becoming an NPL.
- Generate an NPL probability for each loan record.
- Classify loans into Low, Medium, and High risk categories.
- Analyze loan portfolio risk using Power BI.
- Provide business insights that can support credit-risk monitoring and decision-making.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- Pickle
- Power BI
- Microsoft Excel
- Streamlit

## 🤖 Machine Learning

The project uses a machine learning pipeline to process loan information and generate NPL predictions.

The trained Random Forest model is saved as:

`models/random_forest_npl_model.pkl`

The model generates:

1. NPL Prediction
2. NPL Probability
3. Risk Level

### Risk Classification

| NPL Probability | Risk Level |
|---|---|
| Below 30% | Low Risk |
| 30% – 69% | Medium Risk |
| 70% and above | High Risk |

## 📊 Power BI Dashboard

The Power BI dashboard provides an interactive view of the loan portfolio and NPL risk.

The dashboard can be used to analyze:

- Loan amounts
- NPL performance
- Risk levels
- Loan types
- Lenders
- Locations
- Agents
- Market performance
- Other portfolio-level risk indicators

## 📈 Project Workflow

```text
Loan Data
    ↓
Data Cleaning & Preparation
    ↓
Feature Engineering
    ↓
Machine Learning Model
    ↓
NPL Prediction
    ↓
NPL Probability
    ↓
Risk Classification
    ↓
Power BI Dashboard
    ↓
Business Insights