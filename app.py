import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit App
st.title("📊 Bank Marketing Campaign Prediction")
st.write("This app predicts if a client will subscribe to a term deposit.")

# User Input Fields
age = st.number_input("Age", min_value=18, max_value=80, value=30)

job = st.selectbox(
    "Job Type", 
    ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", 
     "retired", "self-employed", "services", "student", "technician", 
     "unemployed", "unknown"]
)

marital = st.selectbox("Marital Status", ["married", "single", "divorced"])

education = st.selectbox("Education Level", ["primary", "secondary", "tertiary", "unknown"])

default = st.radio("Has Credit in Default?", ["yes", "no"])

balance = st.number_input("Account Balance", min_value=-10000, max_value=100000, value=0)

housing = st.radio("Has Housing Loan?", ["yes", "no"])

loan = st.radio("Has Personal Loan?", ["yes", "no"])

contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])

day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=15)

month = st.selectbox("Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun", 
                                            "jul", "aug", "sep", "oct", "nov", "dec"])

duration = st.number_input("Contact Duration (seconds)", min_value=0, max_value=5000, value=100)

campaign = st.number_input("Number of Contacts in this Campaign", min_value=1, max_value=50, value=1)

pdays = st.number_input("Days Passed After Last Contact (-1 means never contacted)", min_value=-1, max_value=500, value=-1)

previous = st.number_input("Number of Contacts Before this Campaign", min_value=0, max_value=50, value=0)

poutcome = st.selectbox("Outcome of Previous Campaign", ["failure", "success", "other", "unknown"])

# # Convert categorical inputs to numerical values (if required)
# def preprocess_inputs():
#     job_mapping = {"admin.": 0, "blue-collar": 1, "entrepreneur": 2, "housemaid": 3, 
#                    "management": 4, "retired": 5, "self-employed": 6, "services": 7, 
#                    "student": 8, "technician": 9, "unemployed": 10, "unknown": 11}
    
#     marital_mapping = {"married": 0, "single": 1, "divorced": 2}
    
#     education_mapping = {"primary": 0, "secondary": 1, "tertiary": 2, "unknown": 3}
    
#     default_mapping = {"yes": 1, "no": 0}
    
#     housing_mapping = {"yes": 1, "no": 0}
    
#     loan_mapping = {"yes": 1, "no": 0}
    
#     contact_mapping = {"cellular": 0, "telephone": 1, "unknown": 2}
    
#     month_mapping = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, 
#                      "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
    
#     poutcome_mapping = {"failure": 0, "success": 1, "other": 2, "unknown": 3}
    
# return np.array([
#     age, 
#     job_mapping[job], 
#     marital_mapping[marital], 
#     education_mapping[education], 
#     default_mapping[default], 
#     balance, 
#     housing_mapping[housing], 
#     loan_mapping[loan], 
#     contact_mapping[contact], 
#     day, 
#     month_mapping[month], 
#     duration, 
#     campaign, 
#     pdays, 
#     previous, 
#     poutcome_mapping[poutcome]]).reshape(1, -1)

# Make prediction when the button is clicked
if st.button("Predict"):
    input_data = preprocess_inputs()
    prediction = model.predict(input_data)
    
    result = "✅ Client is likely to subscribe to a term deposit." if prediction[0] == 1 else "❌ Client is NOT likely to subscribe."

    st.success(result)
