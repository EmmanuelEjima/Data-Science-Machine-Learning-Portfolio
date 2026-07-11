# ⚕️ Breast Cancer Diagnostic Assistant

An end-to-end Machine Learning web application that predicts whether a breast tumor is **Malignant** or **Benign** using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset. The project demonstrates the complete machine learning lifecycle, from data preprocessing and model training to cloud deployment with **Streamlit Community Cloud**.

## 🌐 Live Demo

🚀 **Streamlit Application**  
https://breast-cancer-diagnostic-assistant.streamlit.app/

📓 **Google Colab Notebook**  
https://colab.research.google.com/drive/1NrM_aeOKVae3eZfqY3c0B64S1PF4pckD?usp=sharing



## 📌 Project Overview

This project implements a standardized machine learning pipeline for breast cancer diagnosis. It allows users to input **30 clinical features** extracted from Fine Needle Aspirate (FNA) images of breast masses and instantly predicts whether the tumor is **Malignant** or **Benign**.

The application demonstrates best practices in machine learning model development, reproducibility, and deployment.



## ✨ Features

- Interactive Streamlit web application
- Complete 30-feature clinical input form
- Real-time prediction (Malignant or Benign)
- Logistic Regression classifier
- StandardScaler preprocessing pipeline
- Serialized model using Joblib
- Cloud deployment with Streamlit Community Cloud
- Clean and responsive user interface



## 📊 Machine Learning Workflow

- Data Loading
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Scaling using StandardScaler
- Model Training
- Cross Validation
- Model Evaluation
- Model Serialization (.pkl)
- Streamlit Deployment



## 📈 Model Performance

| Metric | Value |
|---------|-------|
| Algorithm | Logistic Regression |
| Dataset | Wisconsin Diagnostic Breast Cancer (WDBC) |
| Cross-Validation Accuracy | **≈98.1%** |
| Framework | Scikit-learn |


## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Google Colab
- Git
- GitHub



## 📂 Project Structure

```text
breast-cancer-diagnostic-app/
│
├── app.py
├── README.md
├── requirements.txt
└── models/
    ├── breast_cancer_model.pkl
    └── scaler.pkl
```



## ▶️ Run the Project Locally

Clone the repository

```bash
git clone https://github.com/EmmanuelEjima/Data-Science-Machine-Learning-Portfolio.git
```

Navigate to the project directory

```bash
cd projects/breast-cancer-diagnostic-app
```

Install the required packages

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```



## 📚 Dataset

**Wisconsin Diagnostic Breast Cancer (WDBC)**

The dataset contains measurements computed from digitized images of Fine Needle Aspirate (FNA) of breast masses. Each observation includes 30 numerical features describing cell nuclei characteristics and is classified as either:

- **Malignant (Cancerous)**
- **Benign (Non-cancerous)**



## 📌 Future Improvements

- Display prediction probability/confidence score
- Interactive data visualizations
- Feature importance analysis
- Additional machine learning model comparison
- Model explainability using SHAP


## ⚠️ Disclaimer

This application is intended solely for educational, research, and portfolio purposes. It is **not a medical diagnostic tool** and should not replace professional medical advice or clinical decision-making.



## 👨‍💻 Author

**Emmanuel Ejima**

Chemical Engineer | Data Science & Machine Learning Enthusiast | Renewable Energy Engineer

- **GitHub:** https://github.com/EmmanuelEjima
- **LinkedIn:** 



⭐ If you found this project useful, consider giving it a **Star** on GitHub.
