import os
import pickle
import streamlit as st
import pandas as pd
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Metaverse Fraud Analysis",
                   layout="wide",
                   page_icon="🔐")


# Load the saved models
with open('rf_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.sav', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('encoder.sav', 'rb') as encoder_file:
    encoder_columns = pickle.load(encoder_file)

# Define the title text
title_text = "Metaverse Fraud Detection"

# Define the background color and text color of the title box
background_color = "#23395d"
box_background_color = "#23395d"
text_color = "#ffffff"

# Apply HTML and CSS to style the title
title_html = f"""
    <div style="background-color:{box_background_color};padding:8px;border-radius:10px;">
        <h1 style="color:{text_color};text-align:center;">{title_text}</h1>
    </div>
    <body>
      <br>
        <center>Welcome! We're here to safeguard the integrity of virtual worlds and ensure a fraud-free experience.</center>
    </body>
"""

# Display the styled title using markdown
st.markdown(title_html, unsafe_allow_html=True)

# Getting the input data from the user
col1, col2 = st.columns(2)

with col1:
    month = st.selectbox('Transaction Month',('1','2','3','4','5','6','7','8','9','10','11','12'))
with col2:
    day_of_week = st.selectbox('Transaction Day',('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'))
with col1:
    time_category = st.selectbox('Transaction Time',('before 6am', '6am to 11.59pm', '12pm to 6pm','after 6pm'))
with col2:
    amount = st.number_input('Transaction Amount')
with col1:
    transaction_type = st.selectbox('Transaction Type', ('phishing', 'purchase', 'sale', 'scam', 'transfer'))
with col2:
    session_duration = st.number_input('Time taken of each activity sessions (minutes)')
with col1:
    login_frequency = st.number_input('Login Frequency',step=1)
with col2:
    risk_score = st.number_input('Risk Score')
with col1:
    location_region = st.selectbox('Transaction Location', ('Africa', 'Asia', 'Europe', 'North America', 'South America'))
with col2:
    ip_prefix = st.selectbox('IP Prefix', ('10.0', '172.0', '172.16', '192.0', '192.168'))
with col1:
    purchase_pattern = st.selectbox('Pattern of Purchases Behaviour', ('focused', 'high value', 'random'))
with col2:
    age_group = st.selectbox('Age Group Based on Activity History', ('new', 'veteran', 'established'))

# Code for Prediction
if st.button('Predict Risk'):
    # Create a DataFrame with input data
    input_data = pd.DataFrame({
        'amount': [amount],
        'login_frequency': [login_frequency],
        'session_duration': [session_duration],
        'risk_score': [risk_score],
        'month':[month],
        'day_of_week': [day_of_week],
        'time_category': [time_category],
        'transaction_type': [transaction_type],
        'location_region': [location_region],
        'ip_prefix': [ip_prefix],
        'purchase_pattern': [purchase_pattern],
        'age_group': [age_group]
    })

    # Perform one-hot encoding using pd.get_dummies
    input_data = pd.get_dummies(input_data[['month','day_of_week','time_category','transaction_type', 'location_region', 'ip_prefix', 'purchase_pattern', 'age_group']], dtype=int)
    
    # Reorder columns to match training data
    input_data = input_data.reindex(columns=encoder_columns, fill_value=0)
    
    # Scale the numerical features
    numerical_features = ['amount', 'login_frequency', 'session_duration', 'risk_score']

    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

    # Make the prediction
    prediction = model.predict(input_data)
    # Display the prediction result

    if prediction[0] == '0':
        st.success('This transaction is predicted to be low risk.')
    elif prediction[0] == '1':
        st.warning('This transaction is predicted to be moderate risk.')
    elif prediction[0] == '2':
        st.error('This transaction is predicted to be high risk.')
