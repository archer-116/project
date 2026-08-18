"""
train_models.py
----------------
Trains 5 classification models on the Bank Marketing (Portuguese bank
term-deposit) dataset, evaluates them with 6 metrics, saves the trained
models + preprocessing objects for the Streamlit app, and writes a
test_data.csv sample that will be used for the Streamlit demo / upload.

Dataset: Bank Marketing (UCI ML Repository)
Source used in this script: https://raw.githubusercontent.com/selva86/datasets/master/bank-full.csv
(This is the "bank-additional-full" version of the dataset: 41,188 rows, 20 features, binary target 'y')

Run this once (locally / on BITS Virtual Lab / Colab) to produce the `model/` folder
that app.py needs.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42
DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/bank-full.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_URL, sep=";")
df.columns = [c.strip() for c in df.columns]
print(f"Shape: {df.shape}")
print(df.head())

TARGET = "y"

# ---------------------------------------------------------------
# 2. Basic cleaning
# ---------------------------------------------------------------
# 'duration' is known to leak the target (call duration is only known AFTER
# the call ends, and is highly predictive of the outcome) -> drop it for a
# realistic, deployable model.
if "duration" in df.columns:
    df = df.drop(columns=["duration"])

df = df.drop_duplicates().reset_index(drop=True)

categorical_cols = df.select_dtypes(include="object").columns.tolist()
categorical_cols.remove(TARGET)
numeric_cols = [c for c in df.columns if c not in categorical_cols + [TARGET]]

print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")

# ---------------------------------------------------------------
# 3. Encode target and features
# ---------------------------------------------------------------
target_le = LabelEncoder()
y = target_le.fit_transform(df[TARGET])  # no=0, yes=1

# One-hot encode categorical features (keeps things simple & robust
# for a Streamlit app where users upload raw-looking test rows)
X = pd.get_dummies(df[categorical_cols + numeric_cols],
                    columns=categorical_cols, drop_first=True)

feature_columns = X.columns.tolist()
print(f"Total features after encoding: {len(feature_columns)}")

# ---------------------------------------------------------------
# 4. Train / test split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------
# 5. Scale numeric features (needed for LR / KNN; harmless for tree models)
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 6. Save a test_data.csv (RAW, un-encoded rows + true label) for the
#    assignment submission and for uploading into the Streamlit app.
# ---------------------------------------------------------------
test_idx = X_test.index
test_data_raw = df.loc[test_idx].copy()
test_data_raw.to_csv("test_data.csv", index=False)
print(f"Saved test_data.csv with shape {test_data_raw.shape}")

# ---------------------------------------------------------------
# 7. Define models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
}

# ---------------------------------------------------------------
# 8. Train, evaluate, save
# ---------------------------------------------------------------
results = {}
for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = y_pred

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    results[name] = metrics
    print({k: round(v, 4) for k, v in metrics.items()})

    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, os.path.join(MODEL_DIR, f"{fname}.pkl"))

# ---------------------------------------------------------------
# 9. Save shared preprocessing objects + results table
# ---------------------------------------------------------------
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(target_le, os.path.join(MODEL_DIR, "target_encoder.pkl"))
joblib.dump(feature_columns, os.path.join(MODEL_DIR, "feature_columns.pkl"))
joblib.dump(categorical_cols, os.path.join(MODEL_DIR, "categorical_cols.pkl"))
joblib.dump(numeric_cols, os.path.join(MODEL_DIR, "numeric_cols.pkl"))

results_df = pd.DataFrame(results).T
results_df.index.name = "ML Model Name"
results_df = results_df.round(4)
results_df.to_csv(os.path.join(MODEL_DIR, "results_comparison.csv"))

print("\n================ FINAL COMPARISON TABLE ================")
print(results_df.to_string())
print("==========================================================")

with open(os.path.join(MODEL_DIR, "results_comparison.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nAll models, preprocessing objects and results saved inside 'model/'")
print("Done.")
