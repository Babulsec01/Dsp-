
import streamlit as st
import pandas as pd
import joblib

st.title('Laptop Price Prediction App')

st.sidebar.header('User Input Parameters')

# Define input fields (adjust based on your dataset features)
brand = st.sidebar.selectbox('Brand', ['Dell', 'HP', 'Apple', 'Asus', 'Lenovo', 'Acer'])
type_name = st.sidebar.selectbox('Type', ['Ultrabook', 'Gaming', 'Notebook', '2 in 1 Convertible'])
ram = st.sidebar.slider('RAM (GB)', 2, 64, 8)
touchscreen = st.sidebar.selectbox('Touchscreen', ['Yes', 'No'])
ips_panel = st.sidebar.selectbox('IPS Panel', ['Yes', 'No'])
ppi = st.sidebar.number_input('PPI (Pixels Per Inch)', min_value=50.0, max_value=400.0, value=150.0)
cpu_brand = st.sidebar.selectbox('CPU Brand', ['Intel', 'AMD'])
gpu_brand = st.sidebar.selectbox('GPU Brand', ['Nvidia', 'AMD', 'Intel'])
hdd = st.sidebar.number_input('HDD (GB)', min_value=0, max_value=2000, value=500)
ssd = st.sidebar.number_input('SSD (GB)', min_value=0, max_value=2000, value=256)
os = st.sidebar.selectbox('Operating System', ['Windows', 'MacOS', 'Linux', 'Other'])

# Prepare input data
input_data = {
    'Brand': brand,
    'TypeName': type_name,
    'Ram': ram,
    'Touchscreen': 1 if touchscreen == 'Yes' else 0,
    'IPS': 1 if ips_panel == 'Yes' else 0,
    'PPI': ppi,
    'Cpu Brand': cpu_brand,
    'Gpu Brand': gpu_brand,
    'HDD': hdd,
    'SSD': ssd,
    'OS': os
}

input_data_df = pd.DataFrame([input_data])

# Load the trained model
model = joblib.load('model_with_pipeline.pkl')  

# Make prediction
result = model.predict(input_data_df)

st.table(input_data_df)
st.metric('Predicted Laptop Price', f'${result[0]:,.2f}')
