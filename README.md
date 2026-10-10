# Credit Card Fraud Detection

An end-to-end machine learning project that predicts the likelihood of credit card transactions being fraudulent. It includes data preprocessing, model evaluation, a REST API, and an interactive Streamlit dashboard.

## Features

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Random Forest classification
- Evaluation using Precision, Recall, F1-score, PR-AUC, and ROC-AUC
- FastAPI prediction endpoint
- Interactive Streamlit dashboard
- Transaction prediction history for the current session
- Model performance metrics and confusion matrix

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Explainability:** SHAP
- **Backend API:** FastAPI, Uvicorn
- **Dashboard:** Streamlit
- **Testing:** Pytest

## Model Performance

Evaluated on a held-out test set of 56,746 transactions.

| Metric | Result |
|---|---:|
| Accuracy | 99.95% |
| Precision | 95.83% |
| Recall | 72.63% |
| F1-score | 82.63% |
| PR-AUC | 81.49% |
| ROC-AUC | 94.26% |

### Confusion Matrix

| Actual Class | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Legitimate | 56,648 | 3 |
| Fraud | 26 | 69 |

**Important:** The model missed 26 of the 95 fraudulent transactions in the test set. Accuracy alone is not sufficient for evaluating fraud detection because the dataset is highly imbalanced. These results are from a historical test set and do not guarantee future performance.

## Project Structure

```text
credit-card-fraud-detection/
├── api/
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── fraud_detection_model.pkl
│   ├── scaler.pkl
│   └── model_metrics.json
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── app.py
├── test_api_pytest.py
└── README.md
```

## Setup and Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd credit-card-fraud-detection
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

Install the packages listed in your project's `requirements.txt`. If you have not created this file yet, that is the next setup task.

### 4. Start the API

From the project root:

```bash
python -m uvicorn api.main:app --reload
```

API documentation:

`http://127.0.0.1:8000/docs`

### 5. Start the Streamlit dashboard

Open a second terminal, activate the virtual environment, and run:

```bash
python -m streamlit run app.py
```

## Dataset

The project uses transaction time, transaction amount, anonymized PCA features (`V1`–`V28`), and a binary fraud label (`Class`).

The raw dataset contains sensitive-looking transaction records and should not be committed to GitHub unnecessarily. Follow the dataset's original license and sharing terms.

## Limitations

- The dataset is highly imbalanced.
- Features `V1`–`V28` are anonymized and do not have straightforward business interpretations.
- The model can miss fraudulent transactions and produce false alerts.
- The application is a portfolio project, not a production-ready payment security system.

## Future Improvements

- Tune the fraud classification threshold.
- Compare additional models and class-imbalance techniques.
- Add input validation and more automated tests.
- Deploy the API and dashboard.
- Monitor model performance and data drift.
