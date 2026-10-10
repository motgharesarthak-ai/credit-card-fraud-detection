import json
from pathlib import Path
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="FraudGuard | Transaction Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000/predict"

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.5rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #10243A, #234B68);
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        color: white;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #D7E6F2;
        margin-bottom: 0;
    }
    .metric-card {
        padding: 1rem;
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 12px;
        min-height: 100px;
    }
    .small-note {
        color: #777;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("💳 FraudGuard")
    st.caption("Transaction Intelligence Platform")
    st.divider()

    st.subheader("Navigation")
    page = st.radio(
        "Choose a page",
        ["Analyze Transaction", "Prediction History", "About Project"],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("Powered by Random Forest + FastAPI")
    st.caption("Academic / portfolio demonstration")

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>Credit Card Fraud Detection</h1>
    <p>Machine learning-based transaction risk assessment</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Demonstration only. Predictions are estimates, not confirmation "
    "of fraud. Do not use this app to make real financial decisions."
)

# =========================================================
# PAGE 1: ANALYZE TRANSACTION
# =========================================================
if page == "Analyze Transaction":

    st.subheader("Analyze a Transaction")
    st.write(
        "Enter the transaction's Time, Amount, and anonymized V1–V28 "
        "features from the dataset."
    )

    with st.form("transaction_form"):
        st.markdown("#### Transaction Information")
        col1, col2 = st.columns(2)

        with col1:
            time_value = st.number_input(
                "Time (seconds)",
                min_value=0.0,
                value=10000.0,
                step=1.0
            )

        with col2:
            amount_value = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=100.0,
                step=1.0
            )

        st.markdown("#### Anonymized Model Features")
        st.caption(
            "For meaningful predictions, use real feature values from "
            "the dataset. The default zeros are placeholders."
        )

        features = {}

        for start in range(1, 29, 4):
            cols = st.columns(4)
            for offset, col in enumerate(cols):
                feature_num = start + offset
                if feature_num <= 28:
                    with col:
                        features[f"V{feature_num}"] = st.number_input(
                            f"V{feature_num}",
                            value=0.0,
                            format="%.6f",
                            key=f"v{feature_num}"
                        )

        submitted = st.form_submit_button(
            "🔍 Analyze Transaction",
            type="primary",
            use_container_width=True
        )

    if submitted:
        transaction = {
            "Time": time_value,
            "Amount": amount_value,
            **features
        }

        with st.spinner("Analyzing transaction..."):
            try:
                response = requests.post(
                    API_URL,
                    json=transaction,
                    timeout=15
                )

                if response.status_code == 200:
                    result = response.json()
                    probability = float(result["fraud_probability"])
                    prediction = int(result["prediction"])

                    st.divider()
                    st.subheader("Prediction Result")

                    result_col, probability_col = st.columns(2)

                    with result_col:
                        if prediction == 1:
                            st.error("⚠️ Potentially Fraudulent")
                            risk_label = "Potentially Fraudulent"
                        else:
                            st.success("✅ Predicted Legitimate")
                            risk_label = "Predicted Legitimate"

                    with probability_col:
                        st.metric(
                            "Estimated Fraud Probability",
                            f"{probability:.2%}"
                        )
                        st.progress(
                            min(max(probability, 0.0), 1.0)
                        )

                    st.caption(
                        "The probability is a model estimate and may "
                        "not be calibrated to real-world fraud rates."
                    )

                    # Save result in session history
                    st.session_state.history.insert(0, {
                        "Timestamp": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "Amount": amount_value,
                        "Prediction": risk_label,
                        "Fraud Probability": probability
                    })

                else:
                    st.error(
                        f"API error ({response.status_code}): "
                        f"{response.text}"
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to FastAPI. Start the backend with: "
                    "python -m uvicorn api.main:app --reload"
                )

            except requests.exceptions.Timeout:
                st.error("The API request timed out. Try again.")

            except (requests.exceptions.RequestException, ValueError, KeyError) as error:
                st.error(f"Could not process the prediction: {error}")

# =========================================================
# PAGE 2: PREDICTION HISTORY
# =========================================================
elif page == "Prediction History":

    st.subheader("Prediction History")
    st.write("Predictions made during this browser session.")

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)

        total = len(history_df)
        fraud_count = (
            history_df["Prediction"] == "Potentially Fraudulent"
        ).sum()
        legitimate_count = total - fraud_count

        c1, c2, c3 = st.columns(3)
        c1.metric("Transactions Analyzed", total)
        c2.metric("Predicted Fraud", int(fraud_count))
        c3.metric("Predicted Legitimate", int(legitimate_count))

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download History as CSV",
            data=csv_data,
            file_name="fraud_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info(
            "No predictions yet. Go to Analyze Transaction to get started."
        )

# =========================================================
# PAGE 3: ABOUT PROJECT
# =========================================================

elif page == "About Project":

    st.subheader("About This Project")

    st.write(
        "An end-to-end machine learning application that estimates "
        "the likelihood of potentially fraudulent credit card transactions."
    )

    st.markdown("#### Technology Stack")
    st.markdown("""
    - **Python and Pandas** — data processing
    - **Scikit-learn** — Random Forest classification
    - **SHAP** — model explainability
    - **FastAPI** — prediction API
    - **Streamlit** — interactive dashboard
    """)

    # Load actual evaluation metrics from JSON
    metrics_path = (
        Path(__file__).resolve().parent
        / "models"
        / "model_metrics.json"
    )

    st.markdown("#### Model Performance")

    try:
        with open(metrics_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        st.caption(
            f"Model: {metrics['model_name']} | "
            f"Evaluation: {metrics['evaluation_dataset']} | "
            f"Test samples: {metrics['test_samples']:,}"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Precision", f"{metrics['precision']:.2%}")
        col2.metric("Recall", f"{metrics['recall']:.2%}")
        col3.metric("F1-score", f"{metrics['f1_score']:.2%}")

        col4, col5, col6 = st.columns(3)

        col4.metric("PR-AUC", f"{metrics['pr_auc']:.2%}")
        col5.metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")
        col6.metric("Accuracy", f"{metrics['accuracy']:.2%}")

        st.markdown("#### Confusion Matrix")

        
        st.markdown("#### Confusion Matrix")
        cm = metrics["confusion_matrix"]

        st.markdown("#### Test Set Prediction Summary")

        chart_data = pd.DataFrame({
            "Outcome": [
                "Correctly identified fraud",
                "Missed fraud",
                "False fraud alerts",
                "Correctly identified legitimate"
            ],
            "Transactions": [
                cm["true_positives"],
                cm["false_negatives"],
                cm["false_positives"],
                cm["true_negatives"]
            ]
        })

        st.bar_chart(
            chart_data.set_index("Outcome"),
            y="Transactions"
        )

        confusion_df = pd.DataFrame(
            [
                [cm["true_negatives"], cm["false_positives"]],
                [cm["false_negatives"], cm["true_positives"]]
            ],
            index=["Actual Legitimate", "Actual Fraud"],
            columns=["Predicted Legitimate", "Predicted Fraud"]
        )


        st.dataframe(
    confusion_df.style.background_gradient(cmap="Blues"),
    use_container_width=True
)

        st.caption(
            "The model correctly identified "
            f"{cm['true_positives']} fraud cases and missed "
            f"{cm['false_negatives']} fraud cases in the test set."
        )

    except FileNotFoundError:
        st.error(
            "Metrics file not found. Please create "
            "models/model_metrics.json first."
        )

    st.info(
        "These metrics represent historical held-out test performance. "
        "They do not guarantee future performance or establish that "
        "a particular live transaction is fraudulent."
    )

    st.markdown("#### Dataset")
    st.write(
        "The dataset contains transaction time, amount, anonymized "
        "PCA features V1–V28, and a fraud label."
    )
