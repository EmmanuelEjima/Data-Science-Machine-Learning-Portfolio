import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title='Breast Cancer Diagnostics', layout='wide')
st.title('⚕  Breast Cancer Diagnostic Assistant')

@st.cache_resource
def load_assets():
    model = joblib.load('models/breast_cancer_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    st.sidebar.success('Model Loaded')
except Exception as e:
    st.sidebar.error(f'Error: {e}')

radius_mean = st.number_input('Radius Mean', value=14.12)
texture_mean = st.number_input('Texture Mean', value=19.29)
perimeter_mean = st.number_input('Perimeter Mean', value=91.96)
area_mean = st.number_input('Area Mean', value=654.80)

if st.button('Predict'):
    user_features = np.zeros(30)
    user_features[0] = radius_mean
    user_features[1] = texture_mean
    user_features[2] = perimeter_mean
    user_features[3] = area_mean
    scaled = scaler.transform(user_features.reshape(1, -1))
    pred = model.predict(scaled)
    st.write('Malignant' if pred[0] == 0 else 'Benign')
