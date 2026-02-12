import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('fraud_detection_model.pkl')

st.title("Fraud Detection Application")

st.markdown("Please enter the transaction details to predict if it's fraudulent or not.")

st.divider()

# Collect user input
transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
amount = st.number_input("Transaction Amount", min_value=0.0, value = 2000.0)
oldbalanceOrig = st.number_input("Old Balance (Sender)", min_value=0.0, value = 50000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value = 90000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value = 0.0) 
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value = 0.0)

if st.button("Predict"):# Create a DataFrame for the input data 
    input_data = pd.DataFrame([{
        'type': transaction_type, 
         'amount': amount, 
         'oldbalanceOrg': oldbalanceOrig, 
         'newbalanceOrig': newbalanceOrig, 
         'oldbalanceDest': oldbalanceDest, 
         'newbalanceDest': newbalanceDest 
         }])
    # Make prediction
    prediction = model.predict(input_data)[0]

    st.subheader(f"Prediction Result : '{int(prediction)}'" )


    if prediction == 1:
        st.error("The transaction is predicted to be fraudulent.")
    else:
        st.success("The transaction is predicted to be legitimate.")



