import streamlit as st
import numpy as np
import joblib
from pickle import load

# Load the trained model
model = joblib.load("model.pkl")

# Load label encoders
LabelEncoders = load(open("label_encoders.pkl", "rb"))
gender_encoder = LabelEncoders['gender']

st.title("🧠 Stroke Prediction App")
st.markdown("This app predicts whether a person is at risk of stroke based on health information.")

# Input form
gender = st.selectbox("Select Gender", gender_encoder.classes_)
age = st.slider("Age", 1, 100, 45)
hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])
glucose = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)

# Encode gender
gender_encoded = gender_encoder.transform([gender])[0]

# Predict
if st.button("Predict"):
    input_data = np.array([[gender_encoded, age, hypertension, heart_disease, glucose]])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("⚠️ Stroke Risk Detected")
    else:
        st.success("✅ No Stroke Risk")
