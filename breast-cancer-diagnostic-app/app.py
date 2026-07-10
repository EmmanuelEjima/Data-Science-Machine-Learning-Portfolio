import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Breast Cancer Diagnostics", layout="wide", page_icon="🩺")
st.title("🩺 Breast Cancer Diagnostic Assistant")

@st.cache_resource
def load_assets():
    model = joblib.load('models/breast_cancer_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    st.sidebar.success("🚀 Production Model Loaded")
except:
    st.sidebar.error("❌ Model Assets Missing")

st.subheader("Input Tumor Characteristics")
col1, col2 = st.columns(2)
with col1:
    radius_mean = st.number_input("Radius Mean", min_value=0.0, max_value=40.0, value=14.12)
    texture_mean = st.number_input("Texture Mean", min_value=0.0, max_value=50.0, value=19.29)
with col2:
    perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, max_value=250.0, value=91.96)
    area_mean = st.number_input("Area Mean", min_value=0.0, max_value=2500.0, value=654.80)

if st.button("Run Diagnostic Analysis", type="primary"):
    # Model expects 30 features; we pad the inputs to match training dimensions
    user_features = np.zeros(30)
    user_features[0] = radius_mean
    user_features[1] = texture_mean
    user_features[2] = perimeter_mean
    user_features[3] = area_mean
    
    scaled_features = scaler.transform(user_features.reshape(1, -1))
    prediction = model.predict(scaled_features)
    proba = model.predict_proba(scaled_features)
    
    st.markdown("---")
    if prediction[0] == 0:
        st.error(f"### 🚨 Classification: MALIGNANT ({proba[0][0]*100:.2f}%)")
    else:
        st.success(f"### ✅ Classification: BENIGN ({proba[0][1]*100:.2f}%)")
