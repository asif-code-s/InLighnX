# 📊 InLighnX – Telco Customer Churn Prediction

This project predicts whether a customer will churn based on their demographic and service usage details. It includes a machine learning model, an interactive web app built with Streamlit, and a Jupyter notebook for model development and testing.

---

## 🚀 Features

- 📈 Predict customer churn using a trained ML model.
- 🧠 User-friendly Streamlit web interface for live predictions.
- 📥 Explore and download the customer churn dataset.
- 📊 Customize input fields dynamically in the UI.
- 💡 Clear visual feedback on prediction results with probability.

---

## 🗂️ Project Structure

InLighnX/
├── churn_prediction.ipynb # Jupyter notebook for data preprocessing, training, and evaluation
├── churn_model.pkl # Trained classification model (Joblib format)
├── data/
│ └── customer_churn.csv # Dataset used for training and inference
└── web.py # Streamlit web app for live churn prediction


---

## 🧪 Model Details

The machine learning model is a classifier trained on the [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn). Key features include:

- Categorical encoding using one-hot encoding.
- Input fields: gender, SeniorCitizen, Partner, Dependents, Internet services, Contract type, and more.
- Output: Binary prediction (Churn or Not Churn) with confidence score.

---

## 💻 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/asif-code-s/InLighnX.git
cd InLighnX
```
### 2. Install Dependencies

### 3. Launch the Streamlit App
```bash
streamlit run web.py
```
You will see the app open in your browser.

### 📁 Dataset
The dataset should be placed inside a data/ folder as:

```bash
data/customer_churn.csv
```
You can use any Telco churn dataset with similar structure, or refer to this version on Kaggle.

### ✍️ Author
Asif Hussain A
GitHub: https://github.com/asif-code-s/InLighnX

