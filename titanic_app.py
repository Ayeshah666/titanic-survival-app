import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.pipeline import Pipeline
import joblib

# Load the model and preprocessor
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('titanic_model.h5')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_model()

# App title
st.title('Titanic Survival Predictor')

# User input form
st.header('Passenger Details')
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox('Passenger Class', [1, 2, 3])
    sex = st.selectbox('Sex', ['male', 'female'])
    age = st.number_input('Age', min_value=0, max_value=100, value=30)

with col2:
    sibsp = st.number_input('Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)
    parch = st.number_input('Parents/Children Aboard', min_value=0, max_value=10, value=0)
    fare = st.number_input('Fare Price', min_value=0.0, max_value=600.0, value=50.0)

# Create input DataFrame
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'Siblings/Spouses Aboard': [sibsp],
    'Parents/Children Aboard': [parch],
    'Fare': [fare]
})

# Preprocess and predict
if st.button('Predict Survival'):
    # Preprocess the input
    processed_input = preprocessor.transform(input_data)
    
    # Make prediction
    prediction = model.predict(processed_input)[0][0]
    prediction = np.clip(prediction, 0, 1)  # Ensure between 0 and 1
    
    # Display result
    st.subheader('Prediction Result')
    st.write(f'Survival Probability: {prediction*100:.1f}%')
    
    # Visual indicator
    if prediction > 0.5:
        st.success('This passenger likely survived!')
    else:
        st.error('This passenger likely did not survive.')
    
    # Show raw data
    st.subheader('Input Data')
    st.write(input_data)