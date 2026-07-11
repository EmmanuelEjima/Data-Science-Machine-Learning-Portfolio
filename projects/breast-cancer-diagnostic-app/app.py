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
    st.sidebar.success('Model Loaded Successfully')
except Exception as e:
    st.sidebar.error(f'Error loading assets: {e}')

# Define all 30 feature names expected by the scaler/model
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
input_data = {}

for i, feat in enumerate(feature_names):
    with [col1, col2, col3][i % 3]:
        input_data[feat] = st.number_input(feat.replace(' ', '_').title(), value=0.0, format="%.4f")

if st.button('Run Diagnostic Analysis', use_container_width=True):
    # Prepare input as DataFrame to match training format
    input_df = pd.DataFrame([input_data])

    # Preprocess and Predict
    scaled_features = scaler.transform(input_df)
    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)

    st.divider()
    if prediction[0] == 0:
        st.error(f"### Result: Malignant (Confidence: {probability[0][0]:.2%})")
    else:
        st.success(f"### Result: Benign (Confidence: {probability[0][1]:.2%})")
