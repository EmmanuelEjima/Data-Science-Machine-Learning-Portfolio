import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title='Breast Cancer Diagnostics', layout='wide')
st.title('⚕  Breast Cancer Diagnostic Assistant')

# --- RUNTIME DIAGNOSTICS ---
with st.expander("ᄅ Debug: File System Status"):
    st.write(f"Current Path: {os.getcwd()}")
    st.write("Folder Contents:", os.listdir('.'))
    if os.path.exists('models'):
        st.write("✅ 'models' folder exists")
        st.write("Models folder contents:", os.listdir('models'))
    else:
        st.error("❌ 'models' folder NOT found")

@st.cache_resource
def load_assets():
    model = joblib.load('models/breast_cancer_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    st.sidebar.success('Model Loaded Successfully')
except Exception as e:
    st.sidebar.error(f'Error loading assets: {e}')

feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

st.markdown("### Enter Cell Measurements")
col1, col2, col3 = st.columns(3)
input_data = {feat: st.number_input(feat.replace(' ', '_').title(), value=0.0, format='%.4f') for feat in feature_names}

if st.button('Run Diagnostic Analysis', use_container_width=True):
    input_df = pd.DataFrame([input_data])
    scaled_features = scaler.transform(input_df)
    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)
    st.divider()
    if prediction[0] == 0:
        st.error(f'### Result: Malignant (Confidence: {probability[0][0]:.2%})')
    else:
        st.success(f'### Result: Benign (Confidence: {probability[0][1]:.2%})')
