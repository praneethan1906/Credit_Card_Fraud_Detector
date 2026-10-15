import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide"
)

# Load trained model
model = joblib.load("models/model_pipeline.pkl")

# Application title
st.title("💳 Credit Card Fraud Detection")
st.write("Enter the transaction details to predict whether it is fraudulent.")

st.info(
    "This is an academic ML demonstration. "
    "Predictions are estimates, not proof of fraud."
)

# Input: Transaction time
time = st.number_input(
    "Transaction Time (seconds)",
    min_value=0.0,
    value=10000.0
)

# Input: Transaction amount
amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0
)

st.subheader("PCA Components")

st.write(
    "Enter values for V1 to V28. "
    "For testing, you can use values from your dataset."
)

# Create input fields in 4 columns
pca_values = {}

columns = st.columns(4)

for i in range(1, 29):

    with columns[(i - 1) % 4]:

        pca_values[f"V{i}"] = st.number_input(
            f"V{i}",
            value=0.0,
            key=f"V{i}"
        )

# Prediction button
if st.button("Predict Transaction", type="primary"):

    # Create input data
    input_data = {
        "Time": time,
        **pca_values,
        "Amount": amount
    }

    # Convert input into DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure correct feature order
    feature_order = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount"]
    )

    input_df = input_df[feature_order]

    # Predict transaction class
    prediction = model.predict(input_df)[0]

    # Display result
    if prediction == 1:

        st.error("🚨 Prediction: Potentially Fraudulent Transaction")

    else:

        st.success("✅ Prediction: Legitimate Transaction")

    # Display probability if supported
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_df)[0]

        fraud_probability = probabilities[1] * 100

        st.metric(
            "Estimated Fraud Probability",
            f"{fraud_probability:.2f}%"
        )