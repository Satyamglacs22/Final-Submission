import pickle
import streamlit as st
import numpy as np

# Load models
model_logistic = pickle.load(open('model_diabetes.sav', 'rb'))

# Basic UI
st.title("Diabetes Prediction")
st.subheader("Please Enter the Required Information:")

# Input fields
model_choice = st.selectbox('Select the Model', ['Logistic Regression'])

Pregnancies = st.number_input('Enter the Pregnancies value', min_value=0, step=1, key='Pregnancies')
Glucose = st.number_input('Enter the Glucose value', min_value=0.0, key='Glucose')
BloodPressure = st.number_input('Enter the Blood Pressure value', min_value=0.0, key='BloodPressure')
SkinThickness = st.number_input('Enter the Skin Thickness value', min_value=0.0, key='SkinThickness')
Insulin = st.number_input('Enter the Insulin value', min_value=0.0, key='Insulin')
BMI = st.number_input('Enter the BMI value', min_value=0.0, key='BMI')
DiabetesPedigreeFunction = st.number_input('Enter the Diabetes Pedigree Function value', min_value=0.0, key='DiabetesPedigreeFunction')
Age = st.number_input('Enter the Age value', min_value=0, step=1, key='Age')

# Prediction
diabetes_diagnosis = ''

if st.button('Diabetes Prediction Test', key='predict'):
    # Select the model
    if model_choice == 'Logistic Regression':
        model = model_logistic

    # Prepare input data
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
    try:
        diabetes_prediction = model.predict(input_data)
        if diabetes_prediction[0] == 1:
            diabetes_diagnosis = "The patient has diabetes."
        else:
            diabetes_diagnosis = "The patient does not have diabetes."
    except Exception as e:
        diabetes_diagnosis = f"Error: {str(e)}"

st.write(diabetes_diagnosis)
