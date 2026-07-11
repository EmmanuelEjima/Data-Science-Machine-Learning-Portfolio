import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title='Breast Cancer Diagnostics', layout='wide')
st.title('⚕  Breast Cancer Diagnostic Assistant')

@st.cache_resource
def load_assets():
    BASE_DIR = Path(__file__).parent
    model = joblib.load(BASE_DIR / 'models' / 'breast_cancer_model.pkl')
    scaler = joblib.load(BASE_DIR / 'models' / 'scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    st.sidebar.success('Model Loaded Successfully')
except Exception as e:
    st.sidebar.error(f'Error: {e}')

# ... UI and Prediction logic ...
