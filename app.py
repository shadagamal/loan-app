import streamlit as st
import joblib
import numpy as np

# load model
model = joblib.load("model.pkl")

st.title("🏠 House Price Prediction")

size = st.number_input("House Size (m²)")
bedrooms = st.number_input("Number of Bedrooms")

if st.button("Predict Price"):
    
    prediction = model.predict([[size, bedrooms]])
    
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")
