import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

# -----------------------------
# STEP 1 — Load Model
# -----------------------------
model = joblib.load("lr_model.joblib")

# -----------------------------
# STEP 2 — Streamlit Layout
# -----------------------------

# Sidebar: Project Info only
st.sidebar.header("About This Project")
st.sidebar.info("""
This app predicts the **Profit** of a startup based on R&D Spend, Administration, Marketing Spend, and State.

- Model trained on 50 Startups dataset  
- Input the features below to get a real-time prediction  
- State is encoded internally (California, Florida, New York)
""")

# Main Page: Title
st.title("50 Startups Profit Prediction")
st.subheader("Input Features")

# -----------------------------
# STEP 3 — Feature Inputs (Main Page)
# -----------------------------
rd_spend = st.number_input("R&D Spend ($)", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
admin_spend = st.number_input("Administration ($)", min_value=0.0, value=120000.0, step=1000.0, format="%.2f")
marketing_spend = st.number_input("Marketing Spend ($)", min_value=0.0, value=300000.0, step=1000.0, format="%.2f")
state = st.selectbox("State", ["California", "Florida", "New York"])

# Encode State manually
state_ca = 1 if state == "California" else 0
state_fl = 1 if state == "Florida" else 0

# -----------------------------
# STEP 4 — Prepare Input for Model
# -----------------------------
input_df = pd.DataFrame({
    "R&D Spend": [rd_spend],
    "Administration": [admin_spend],
    "Marketing Spend": [marketing_spend],
    "State_Florida": [state_fl],
    "State_California": [state_ca]
})

# Ensure all columns the model expects are present
for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match training
input_df = input_df[model.feature_names_in_]

# -----------------------------
# STEP 5 — Prediction
# -----------------------------
predicted_profit = model.predict(input_df)[0]

st.subheader("Predicted Profit")
st.success(f"${predicted_profit:,.2f}")

# -----------------------------
# STEP 6 — Optional: Actual vs Predicted Scatter
# -----------------------------
if st.checkbox("Show Sample Scatterplot (Actual vs Predicted)"):
    df = pd.read_csv("50_Startups.csv")
    df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)
    X = df_encoded.drop('Profit', axis=1)
    y = df_encoded['Profit']
    y_pred = model.predict(X)

    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y, y=y_pred, color='blue', alpha=0.4, label='Predicted')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel("Actual Profit")
    plt.ylabel("Predicted Profit")
    plt.title("Actual vs Predicted Profit")
    plt.legend()
    st.pyplot(plt)