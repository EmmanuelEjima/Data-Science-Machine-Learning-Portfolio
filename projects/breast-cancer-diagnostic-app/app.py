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

feature_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']

st.subheader("Enter Patient Data")
cols = st.columns(3)
input_data = {}

for i, feat in enumerate(feature_names):
    with cols[i % 3]:
        input_data[feat] = st.number_input(feat.title(), value=0.0, format='%.4f')

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
