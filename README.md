# 🛡️ Credit Risk Prediction App

This is a machine learning web application that predicts the likelihood of a loan applicant defaulting. The model is built using **XGBoost** and is optimized for **High Recall** to ensure that potential high-risk borrowers are identified effectively.

## 🚀 Live Demo

(https://credit-risk-app-app-jmuhewecffz6swelkzwxuv.streamlit.app/)
## 📊 Project Overview
The goal of this project is to assist financial institutions in assessing the risk of lending to individuals. By analyzing historical data, the model provides a probability of default and a binary "Approved" or "Rejected" decision based on an optimized threshold.

### Key Features:
* **XGBoost Classifier**: A powerful gradient boosting model tuned for credit data.
* **Custom Thresholding**: Optimized to achieve ~85% Recall to minimize lending risk.
* **Feature Engineering**: Includes custom metrics like Loan-to-Income ratio and Loan-per-Age.
* **Interactive UI**: Built with Streamlit for real-time risk assessment.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Category Encoders
* **Deployment:** Streamlit Cloud

## 📂 Project Structure
* `app.py`: The main Streamlit application script.
* `requirements.txt`: List of dependencies required to run the app in the cloud.
* `xgb_model.pkl`: The trained XGBoost model.
* `preprocessor.pkl`: Scikit-Learn pipeline for data scaling and encoding.
* `decision_threshold.pkl`: The saved optimal threshold for classification.

## ⚙️ How to Run Locally
1. Clone this repository:
   ```bash
   git clone [https://github.com/michaelll10/credit-risk-app.git](https://github.com/michaelll10/credit-risk-app.git)
