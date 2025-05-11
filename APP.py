import streamlit as st
import pandas as pd
import joblib
from Preprocessing import full_preprocessing # type: ignore

# Load model
model = joblib.load("Fraud_det.pkl")

st.title("Fraud Detection System")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    processed_data = full_preprocessing(data)
    predictions = model.predict(processed_data)
    data['Prediction'] = predictions

    st.write("Prediction Results:")
    st.dataframe(data)

    st.download_button(
        label="Download Predictions as Excel",
        data=data.to_excel(index=False, engine='openpyxl'),
        file_name='fraud_predictions.xlsx'
    )