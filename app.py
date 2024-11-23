import pickle
import streamlit as st
import numpy as np

# Load models
model_logistic = pickle.load(open('model_diabetes.sav', 'rb'))

# Basic UI
st.title("Diabetes Prediction")
st.write("""
This is a Simple Web app to Predict whether a Person has diabetes or not.To Predict,
Please input the required values and click on the 'Predict' Button.
""")

st.subheader("Please Enter the Required Information:")

# Input fields
model_choice = st.selectbox('Select the Model', ['Logistic Regression'])

Age = st.number_input('Age', min_value=10, max_value=100, step=1)
BMI = st.number_input('BMI', min_value=0.0, max_value=70.0, step=0.1)
Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, step=1)
Glucose = st.number_input('Glucose', min_value=0, max_value=200, step=1)
BloodPressure = st.number_input('Blood Pressure', min_value=20, max_value=200, step=1)
SkinThickness = st.number_input('Skin Thickness', min_value=0, max_value=100, step=1)
Insulin = st.number_input('Insulin', min_value=0, max_value=900, step=1)
DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, step=0.01)

# Prediction
diabetes_diagnosis = ''

if st.button('Predict', key='predict'):
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
