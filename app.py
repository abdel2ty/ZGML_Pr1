import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

# Load Trained Model
model = joblib.load("lr_model.joblib")

# App Title
st.title("50 Startups Profit Prediction")
st.info("This app predicts the **Profit** of a startup based on its R&D Spend, Administration, Marketing Spend, and State.")

# User Inputs
st.sidebar.header("Input Features")

rd_spend = st.sidebar.number_input("R&D Spend ($)", min_value=0.0, value=100000.0)
admin_spend = st.sidebar.number_input("Administration ($)", min_value=0.0, value=120000.0)
marketing_spend = st.sidebar.number_input("Marketing Spend ($)", min_value=0.0, value=300000.0)
state = st.sidebar.selectbox("State", ["California", "Florida", "New York"])

# Encode State manually (same as get_dummies with drop_first=True)
state_ca = 1 if state == "California" else 0
state_fl = 1 if state == "Florida" else 0
# New York is the dropped column

# Create DataFrame for model
input_df = pd.DataFrame({
    "R&D Spend": [rd_spend],
    "Administration": [admin_spend],
    "Marketing Spend": [marketing_spend],
    "State_Florida": [state_fl],
    "State_California": [state_ca]
})

# Ensure all columns that the model expects are present
for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match training
input_df = input_df[model.feature_names_in_]

# Prediction
predicted_profit = model.predict(input_df)[0]

st.subheader("Predicted Profit")
st.success(f"${predicted_profit:,.2f}")

# Optional: Show Actual vs Predicted Scatter (Sample Dataset)
if st.checkbox("Show Sample Scatterplot (Actual vs Predicted)"):
    # Load full dataset for plot
    df = pd.read_csv("50_Startups.csv")
    # Encode State
    df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)
    X = df_encoded.drop('Profit', axis=1)
    y = df_encoded['Profit']
    y_pred = model.predict(X)
    
    plt.figure(figsize=(6,6))
    # Actual points (transparent)
    sns.scatterplot(x=y, y=y_pred, color='blue', alpha=0.4, label='Predicted')
    # Identity line
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel("Actual Profit")
    plt.ylabel("Predicted Profit")
    plt.title("Actual vs Predicted Profit")
    plt.legend()
    st.pyplot(plt)