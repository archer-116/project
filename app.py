"""
Streamlit app - Bank Term-Deposit Subscription Predictor
Assignment 2 - Machine Learning (BITS Pilani WILP, M.Tech AIML/DSE)
Student: Vinay Kumar Nutenki | ID: 2025AC05640

Features implemented (per assignment requirement):
  a. Dataset upload option (CSV)              -> file_uploader below
  b. Model selection dropdown                 -> st.selectbox
  c. Display of evaluation metrics            -> metrics table + cards
  d. Confusion matrix / classification report -> heatmap + text report
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

MODEL_DIR = "model"

st.set_page_config(page_title="Bank Term-Deposit Predictor", layout="wide")

# ---------- Part 1: Load saved artifacts ----------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    target_le = joblib.load(os.path.join(MODEL_DIR, "target_encoder.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    categorical_cols = joblib.load(os.path.join(MODEL_DIR, "categorical_cols.pkl"))
    numeric_cols = joblib.load(os.path.join(MODEL_DIR, "numeric_cols.pkl"))

    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "kNN": "knn.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
    }
    models = {name: joblib.load(os.path.join(MODEL_DIR, fname))
              for name, fname in model_files.items()}

    results_df = pd.read_csv(os.path.join(MODEL_DIR, "results_comparison.csv"),
                              index_col="ML Model Name")
    return scaler, target_le, feature_columns, categorical_cols, numeric_cols, models, results_df


scaler, target_le, feature_columns, categorical_cols, numeric_cols, models, results_df = load_artifacts()

# ---------- Part 2: Preprocessing helper (mirrors train_models.py) ----------
def preprocess(df_raw):
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    y_true = None
    if "y" in df.columns:
        y_true = target_le.transform(df["y"])
        df = df.drop(columns=["y"])

    X = pd.get_dummies(df, columns=[c for c in categorical_cols if c in df.columns],
                        drop_first=True)
    # align columns with training-time feature set
    X = X.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(X)
    return X_scaled, y_true


# ---------- Part 3: Sidebar - dataset upload + model selection ----------
st.sidebar.title("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "a. Upload test data (CSV)", type=["csv"],
    help="Upload the provided test_data.csv, or any CSV with the same raw columns "
         "(optionally include the true 'y' column for evaluation metrics)."
)

model_name = st.sidebar.selectbox(
    "b. Select a Model", list(models.keys())
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Student:** Vinay Kumar Nutenki  \n"
    "**ID:** 2025AC05640  \n"
    "**Dataset:** Bank Marketing (UCI) — term deposit subscription"
)

# ---------- Part 4: Main page ----------
st.title("🏦 Bank Term-Deposit Subscription Predictor")
st.caption("Assignment 2 — Machine Learning | Multiple classification models demo")

tab1, tab2, tab3 = st.tabs(["📊 Predictions & Metrics", "📈 Model Comparison", "ℹ️ About"])

with tab1:
    if uploaded_file is None:
        st.info("👈 Upload the `test_data.csv` file from the sidebar to see predictions "
                "and evaluation metrics for the selected model.")
    else:
        df_raw = pd.read_csv(uploaded_file)
        st.subheader("Preview of uploaded data")
        st.dataframe(df_raw.head(10))

        try:
            X_scaled, y_true = preprocess(df_raw)
        except Exception as e:
            st.error(f"Could not process the uploaded file: {e}")
            st.stop()

        model = models[model_name]
        y_pred = model.predict(X_scaled)
        y_pred_labels = target_le.inverse_transform(y_pred)

        st.subheader(f"c. Predictions using: {model_name}")
        pred_display = df_raw.copy()
        pred_display["Predicted_y"] = y_pred_labels
        st.dataframe(pred_display.head(20))

        # download predictions
        csv_out = pred_display.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download predictions as CSV", csv_out,
                            file_name="predictions.csv", mime="text/csv")

        if y_true is not None:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_scaled)[:, 1]
            else:
                y_proba = y_pred

            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_proba)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)

            st.subheader("c. Evaluation Metrics (on uploaded test data)")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Accuracy", f"{acc:.3f}")
            c2.metric("AUC", f"{auc:.3f}")
            c3.metric("Precision", f"{prec:.3f}")
            c4.metric("Recall", f"{rec:.3f}")
            c5.metric("F1 Score", f"{f1:.3f}")
            c6.metric("MCC", f"{mcc:.3f}")

            st.subheader("d. Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=target_le.classes_, yticklabels=target_le.classes_, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            st.subheader("d. Classification Report")
            report = classification_report(y_true, y_pred, target_names=target_le.classes_)
            st.code(report)
        else:
            st.warning("Uploaded CSV has no 'y' column — showing predictions only "
                       "(evaluation metrics need ground-truth labels).")

with tab2:
    st.subheader("Comparison of all 6 evaluation metrics across models")
    st.caption("(Computed once during training on the held-out test split; see README for full details)")
    st.dataframe(results_df.style.highlight_max(axis=0, color="#c9f0d1"))

    metric_to_plot = st.selectbox("Choose a metric to visualize", results_df.columns)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    results_df[metric_to_plot].sort_values().plot(kind="barh", ax=ax2, color="#4C72B0")
    ax2.set_xlabel(metric_to_plot)
    st.pyplot(fig2)

with tab3:
    st.markdown("""
    ### About this app
    This Streamlit app demonstrates **5 classification models** trained on the
    **Bank Marketing** dataset (UCI Machine Learning Repository) to predict whether
    a client will subscribe to a term deposit.

    **Models implemented:**
    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbors
    4. Gaussian Naive Bayes
    5. Random Forest (Ensemble)

    **How to use:**
    1. Upload the provided `test_data.csv` (or a CSV with the same raw columns) from the sidebar.
    2. Pick a model from the dropdown.
    3. View predictions, evaluation metrics, confusion matrix and classification report.
    4. Switch to the **Model Comparison** tab to see all models side by side.
    """)
