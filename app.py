import streamlit as st
import pickle
import pandas as pd

# load model
model = joblib.load("loan_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Loan Approval Prediction")

age = st.number_input("Age")
income = st.number_input("Income")
credit_score = st.number_input("Credit Score")

gender = st.selectbox("Gender", ["Male","Female"])
occupation = st.selectbox("Occupation", ["Engineer","Teacher","Student","Manager","Accountant"])
education = st.selectbox("Education", ["High School","Bachelor's","Master's"])
marital = st.selectbox("Marital Status", ["Single","Married"])

if st.button("Predict"):

    input_dict = {
        "age":age,
        "income":income,
        "credit_score":credit_score
    }

    df = pd.DataFrame([input_dict])

    df = pd.get_dummies(df)

    df = df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Denied")
