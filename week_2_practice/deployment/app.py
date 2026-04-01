import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="kaushalya7/medical_insurance_model",
    filename="best_medical_insurance_model_v1.joblib"
)
model = joblib.load(model_path)

# Streamlit UI for Insurance Charges Prediction
st.title("Insurance Charges Prediction App")
st.write("""
This application predicts the **medical insurance charges** based on personal and lifestyle details.
Please enter the required information below to get a prediction.
""")

# User input
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'bmi': bmi,
    'children': children,
    'smoker': smoker,
    'region': region
}])

# Prediction
if st.button("Predict Charges"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Estimated Insurance Charges: **${prediction:,.2f}**")
