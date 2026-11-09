# app.py - Wine Quality Prediction System (Section B)
# Easy version: trains the model automatically if wine_model.pkl is missing.
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Folder where this file lives (works on Streamlit Cloud even if the app is in a subfolder)
BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "wine_quality.csv")
MODEL_PATH = os.path.join(BASE, "wine_model.pkl")

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


def get_data():
    # Use the local CSV if it exists, otherwise download it from UCI
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    df = pd.read_csv(URL, sep=";")
    df.to_csv(CSV_PATH, index=False)
    return df


def train_and_save():
    # Same steps as train_model.py
    df = get_data().drop_duplicates().copy()
    y = (df["quality"] >= 6).astype(int)
    X = df.drop(columns=["quality"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    m = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced"))
    ])
    m.fit(X_train, y_train)
    joblib.dump(m, MODEL_PATH)
    return m

st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)


@st.cache_resource
def load_model():
    # Load the model; if the file is missing (or unreadable), train a new one
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return train_and_save()


model = load_model()
data = get_data()

st.title("🍷 Wine Quality Prediction System")
st.caption("Section B classroom prototype for preliminary quality classification")

# Basic dataset information
c1, c2, c3 = st.columns(3)
c1.metric("Records", len(data))
c2.metric("Average quality score", round(data["quality"].mean(), 2))
c3.metric("Good-quality records", int((data["quality"] >= 6).sum()))

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Predict uploaded wine samples")
st.write("Upload a CSV containing the same predictor columns as the training data.")

# Sample file the user can download and upload back
template = data.drop(columns=["quality"]).head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False),
    "sample_wine_input.csv",
    "text/csv"
)

uploaded = st.file_uploader("Choose wine sample CSV", type=["csv"])

if uploaded is not None:
    new_data = pd.read_csv(uploaded)

    # Columns the model was trained on
    required_columns = list(model.feature_names_in_)
    missing = [c for c in required_columns if c not in new_data.columns]

    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
    else:
        predictions = model.predict(new_data[required_columns])

        results = new_data.copy()
        results["Prediction"] = [
            "GOOD QUALITY" if value == 1 else "NEEDS IMPROVEMENT"
            for value in predictions
        ]

        st.subheader("Prediction results")
        st.dataframe(results, use_container_width=True)
        st.metric("Predicted good-quality samples", int((predictions == 1).sum()))

        st.download_button(
            "Download prediction results",
            results.to_csv(index=False),
            "wine_quality_predictions.csv",
            "text/csv"
        )

st.info(
    "Responsible use: this is a classroom prototype. A prediction is not a "
    "substitute for laboratory testing, sensory evaluation, food-safety "
    "procedures, or professional quality-control decisions."
)
