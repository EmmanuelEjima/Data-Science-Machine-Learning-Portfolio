import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title='Breast Cancer Diagnostics', layout='wide')
st.title('⚕  Breast Cancer Diagnostic Assistant')

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

# 30 Features paired with their true dataset averages as robust defaults
feature_defaults = {
    'mean radius': 14.1273, 'mean texture': 19.2896, 'mean perimeter': 91.9690, 'mean area': 654.8891,
    'mean smoothness': 0.0964, 'mean compactness': 0.1043, 'mean concavity': 0.0888, 'mean concave points': 0.0489,
    'mean symmetry': 0.1812, 'mean fractal dimension': 0.0628, 'radius error': 0.4052, 'texture error': 1.2169,
    'perimeter error': 2.8661, 'area error': 40.3371, 'smoothness error': 0.0070, 'compactness error': 0.0255,
    'concavity error': 0.0319, 'concave points error': 0.0118, 'symmetry error': 0.0205, 'fractal dimension error': 0.0038,
    'worst radius': 16.2692, 'worst texture': 25.6772, 'worst perimeter': 107.2612, 'worst area': 880.5831,
    'worst smoothness': 0.1324, 'worst compactness': 0.2543, 'worst concavity': 0.2722, 'worst concave points': 0.1146,
    'worst symmetry': 0.2901, 'worst fractal dimension': 0.0839
}

st.subheader("Enter Patient Data")
cols = st.columns(3)
input_data = {}

# Dynamic loop generating form using realistic dataset defaults
for i, (feat, default_val) in enumerate(feature_defaults.items()):
    with cols[i % 3]:
        input_data[feat] = st.number_input(feat.title(), value=float(default_val), format='%.4f')

if st.button('Run Diagnostic Analysis', type='primary'):
    # Convert input dict values straight to a raw array to bypass column naming mismatches
    raw_features = [float(input_data[feat]) for feat in feature_defaults.keys()]
    features_array = np.array(raw_features).reshape(1, -1)
    
    # Scale and predict
    scaled_features = scaler.transform(features_array)
    prediction = model.predict(scaled_features)
    prob = model.predict_proba(scaled_features)
    
    st.divider()
    if prediction[0] == 0:
        st.error(f'### 🚨 Result: Malignant (Confidence: {prob[0][0]:.2%})')
    else:
        st.success(f'### ✅ Result: Benign (Confidence: {prob[0][1]:.2%})')
