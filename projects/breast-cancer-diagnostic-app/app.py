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

st.info('Enter cell measurements to predict diagnosis.')
