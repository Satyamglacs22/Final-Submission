import pickle
import streamlit as st
import numpy as np

# Load models
model_logistic = pickle.load(open('model_diabetes.sav', 'rb'))


# Styling
st.markdown(
    """
    <style>
    body {
        background-image: url("image.png");
        background-size: cover;        
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; color: darkblue;'>Diabetes Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #333399;'>Please Enter the Required Information:</h3>", unsafe_allow_html=True)

# Input fields
model_choice = st.selectbox('Select the Model', ['Logistic Regression', 'Random Forest', 'Gaussian'])

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
    # elif model_choice == 'Random Forest':
        # model = model_random_forest
    # else:
        # model = model_gaussian

    # Prepare input data
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
    try:
        diabetes_prediction = model.predict(input_data)
        if diabetes_prediction[0] == 1:
            diabetes_diagnosis = "<p style='color: red; font-size: 20px;'>The patient has diabetes</p>"
        else:
            diabetes_diagnosis = "<p style='color: green; font-size: 20px;'>The patient does not have diabetes</p>"
    except Exception as e:
        diabetes_diagnosis = f"<p style='color: red; font-size: 20px;'>Error: {str(e)}</p>"

st.markdown(diabetes_diagnosis, unsafe_allow_html=True)
