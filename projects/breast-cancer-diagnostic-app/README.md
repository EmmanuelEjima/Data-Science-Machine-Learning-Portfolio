# ⚕  Breast Cancer Diagnostic Assistant

This is a production-ready machine learning application that predicts whether a tumor is Malignant or Benign based on medical features.

## 🚀 Project Overview
- **Model:** Logistic Regression (Scikit-Learn)
- **Accuracy:** ~98.1% (Cross-Validated)
- **Deployment:** Streamlit Community Cloud

## 📂 Folder Structure
- `app.py`: The main Streamlit application script with 30-feature input.
- `models/`: Contains the serialized model (`breast_cancer_model.pkl`) and scaler (`scaler.pkl`).
- `requirements.txt`: List of Python dependencies.

## 🛠️ Installation & Local Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
