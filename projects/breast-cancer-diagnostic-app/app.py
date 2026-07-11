import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title='Breast Cancer Diagnostics', layout='wide')
st.title('⚕  Breast Cancer Diagnostic Assistant')

@st.cache_resource
def load_assets():
    # Resolve absolute path relative to this file
    BASE_DIR = Path(__file__).parent
    model_path = BASE_DIR / 'models' / 'breast_cancer_model.pkl'
    scaler_path = BASE_DIR / 'models' / 'scaler.pkl'

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_assets()
    st.sidebar.success('Model Loaded Successfully')
except Exception as e:
    st.sidebar.error(f'Error loading assets: {e}')

feature_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']

input_data = {feat: st.number_input(feat.replace(' ', '_').title(), value=0.0, format='%.4f') for feat in feature_names}

if st.button('Run Diagnostic Analysis'):
    input_df = pd.DataFrame([input_data])
    scaled_features = scaler.transform(input_df)
    prediction = model.predict(scaled_features)
    prob = model.predict_proba(scaled_features)
    st.divider()
    if prediction[0] == 0:
        st.error(f'### Result: Malignant (Confidence: {prob[0][0]:.2%})')
    else:
        st.success(f'### Result: Benign (Confidence: {prob[0][1]:.2%})')
