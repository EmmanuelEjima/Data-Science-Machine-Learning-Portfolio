import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Breast Cancer Diagnostic Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Diagnostic Assistant")

st.markdown("""
### AI-Powered Clinical Decision Support System

This application predicts whether a breast tumor is **Malignant** or **Benign**
using a **Logistic Regression** model trained on the
**Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

> **Disclaimer:** This application is for educational and portfolio purposes only.
It should not replace professional medical diagnosis.
""")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_assets():
    BASE_DIR = Path(__file__).parent
    model = joblib.load(BASE_DIR / 'models' / 'breast_cancer_model.pkl')
    scaler = joblib.load(BASE_DIR / 'models' / 'scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.success("✅ Model Loaded Successfully")

    st.markdown("---")

    st.header("Project Information")

    st.write("**Algorithm:** Logistic Regression")
    st.write("**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)")
    st.write("**Cross Validation Accuracy:** 98.1%")
    st.write("**Number of Features:** 30")

    st.markdown("---")

    st.info(
        "Developed by Emmanuel Ejima\n\n"
        "Chemical Engineer | Data Science & Machine Learning"
    )

# ---------------------------------------
# Feature Defaults
# ---------------------------------------

feature_defaults = {
    'mean radius': 14.1273,
    'mean texture': 19.2896,
    'mean perimeter': 91.9690,
    'mean area': 654.8891,
    'mean smoothness': 0.0964,
    'mean compactness': 0.1043,
    'mean concavity': 0.0888,
    'mean concave points': 0.0489,
    'mean symmetry': 0.1812,
    'mean fractal dimension': 0.0628,
    'radius error': 0.4052,
    'texture error': 1.2169,
    'perimeter error': 2.8661,
    'area error': 40.3371,
    'smoothness error': 0.0070,
    'compactness error': 0.0255,
    'concavity error': 0.0319,
    'concave points error': 0.0118,
    'symmetry error': 0.0205,
    'fractal dimension error': 0.0038,
    'worst radius': 16.2692,
    'worst texture': 25.6772,
    'worst perimeter': 107.2612,
    'worst area': 880.5831,
    'worst smoothness': 0.1324,
    'worst compactness': 0.2543,
    'worst concavity': 0.2722,
    'worst concave points': 0.1146,
    'worst symmetry': 0.2901,
    'worst fractal dimension': 0.0839
}

st.markdown("---")

st.header("📋 Patient Clinical Measurements")

st.write(
    "Enter the patient's tumor measurements obtained from a Fine Needle Aspirate (FNA)."
)

cols = st.columns(3)

input_data = {}

for i, (feat, default_val) in enumerate(feature_defaults.items()):
    with cols[i % 3]:
        input_data[feat] = st.number_input(
            feat.title(),
            value=float(default_val),
            format="%.4f"
        )

# ---------------------------------------
# Prediction
# ---------------------------------------

if st.button("🩺 Run Diagnostic Analysis", type="primary"):

    raw_features = [
        float(input_data[feat])
        for feat in feature_defaults.keys()
    ]

    features_array = np.array(raw_features).reshape(1, -1)

    scaled_features = scaler.transform(features_array)

    prediction = model.predict(scaled_features)

    prob = model.predict_proba(scaled_features)

    st.divider()

    confidence = max(prob[0]) * 100

    st.header("📊 Diagnostic Report")

    col1, col2 = st.columns(2)

    if prediction[0] == 0:

        with col1:
            st.metric("Prediction", "🚨 Malignant")

        with col2:
            st.metric("Confidence", f"{confidence:.2f}%")

        st.error("### Clinical Interpretation")

        st.write(
            """
The model detected feature patterns commonly associated with **malignant breast tumors**.

### Recommendation

- Histopathological examination
- Oncology consultation
- Additional diagnostic imaging
- Follow physician recommendations
"""
        )

    else:

        with col1:
            st.metric("Prediction", "✅ Benign")

        with col2:
            st.metric("Confidence", f"{confidence:.2f}%")

        st.success("### Clinical Interpretation")

        st.write(
            """
The model detected feature patterns consistent with **benign breast tumors**.

### Recommendation

- Routine clinical monitoring
- Continue regular medical follow-up
- Follow physician recommendations
"""
        )

# ---------------------------------------
# Footer
# ---------------------------------------

st.markdown("---")

st.caption(
    "🩺 Breast Cancer Diagnostic Assistant | Built with Python, Scikit-learn and Streamlit | © Emmanuel Ejima"
)
