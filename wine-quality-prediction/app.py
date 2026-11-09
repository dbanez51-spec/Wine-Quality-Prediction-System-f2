# app.py - Wine Quality Prediction System (Section B)
# Run with: streamlit run app.py

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "wine_quality.csv"
MODEL_PATH = BASE_DIR / "wine_model.pkl"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
TARGET_COLUMN = "quality"
CLASS_NAMES = ["NEEDS IMPROVEMENT", "GOOD QUALITY"]

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")


@st.cache_data

def get_data():
    """Load the local UCI dataset or download and cache a local copy."""
    if CSV_PATH.exists():
        data = pd.read_csv(CSV_PATH)
    else:
        data = pd.read_csv(DATA_URL, sep=";")
        data.to_csv(CSV_PATH, index=False)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' column.")
    if data.empty:
        raise ValueError("The wine dataset is empty.")
    return data


def train_and_save(data):
    """Train the lab's Random Forest pipeline and save it for later app runs."""
    clean_data = data.drop_duplicates().copy()
    y = (clean_data[TARGET_COLUMN] >= 6).astype(int)
    X = clean_data.drop(columns=[TARGET_COLUMN])
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
        )),
    ])
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


@st.cache_resource

def load_model(data):
    """Load the trained model, training it once if the artifact is not present."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_and_save(data)


def show_evaluation(data, model):
    """Show a reproducible held-out evaluation using the lab's 80/20 split."""
    clean_data = data.drop_duplicates().copy()
    y = (clean_data[TARGET_COLUMN] >= 6).astype(int)
    X = clean_data.drop(columns=[TARGET_COLUMN])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    st.subheader("Model evaluation")
    st.caption(
        "Held-out 20% test split (random_state=42, stratified). "
        "Metrics are illustrative for this dataset and are not a guarantee "
        "of performance on new producers or testing conditions."
    )
    st.metric("Test accuracy", f"{accuracy:.3f}")
    metric_table = pd.DataFrame(report).T
    st.dataframe(
        metric_table.loc[CLASS_NAMES, ["precision", "recall", "f1-score", "support"]],
        use_container_width=True,
    )
    st.write("Confusion matrix (rows = actual class; columns = predicted class):")
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    st.dataframe(
        pd.DataFrame(
            matrix,
            index=[f"Actual {name}" for name in CLASS_NAMES],
            columns=[f"Predicted {name}" for name in CLASS_NAMES],
        ),
        use_container_width=True,
    )


try:
    data = get_data()
    model = load_model(data)
except Exception as exc:
    st.error(f"Could not load the dataset or model: {exc}")
    st.info("Check your internet connection, then run `python train_model.py`.")
    st.stop()

st.title("🍷 Wine Quality Prediction System")
st.caption("Section B classroom prototype for preliminary quality classification")

# Basic dataset information
c1, c2, c3 = st.columns(3)
c1.metric("Records", len(data))
c2.metric("Average quality score", f"{data[TARGET_COLUMN].mean():.2f}")
c3.metric("Good-quality records", int((data[TARGET_COLUMN] >= 6).sum()))

with st.expander("Dataset preview", expanded=True):
    st.dataframe(data.head(10), use_container_width=True)

show_evaluation(data, model)

st.subheader("Predict uploaded wine samples")
st.write(
    "Upload a CSV with the same physicochemical predictor columns used for training. "
    "The original `quality` target column is not an input feature."
)

required_columns = list(model.feature_names_in_)
template = data.reindex(columns=required_columns).head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False),
    "sample_wine_input.csv",
    "text/csv",
)

uploaded = st.file_uploader("Choose wine sample CSV", type=["csv"])
if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not read this CSV. Check that it is a valid, non-empty CSV file. ({exc})")
    else:
        missing = [column for column in required_columns if column not in new_data.columns]
        if new_data.empty:
            st.error("The uploaded CSV contains no sample rows.")
        elif missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            numeric_data = new_data[required_columns].apply(pd.to_numeric, errors="coerce")
            non_numeric = [
                column for column in required_columns
                if (new_data[column].notna() & numeric_data[column].isna()).any()
            ]
            non_finite = [
                column for column in required_columns
                if np.isinf(numeric_data[column].dropna()).any()
            ]
            invalid_columns = sorted(set(non_numeric + non_finite))

            if invalid_columns:
                st.error(
                    "These columns contain non-numeric or infinite values: "
                    + ", ".join(invalid_columns)
                )
            else:
                if numeric_data.isna().any().any():
                    st.warning(
                        "Some measurements are blank. The model pipeline will "
                        "impute them using its training-data medians."
                    )
                try:
                    predictions = model.predict(numeric_data)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                else:
                    results = new_data.copy()
                    results["Prediction"] = [CLASS_NAMES[int(value)] for value in predictions]
                    st.subheader("Prediction results")
                    st.dataframe(results, use_container_width=True)
                    st.metric(
                        "Predicted good-quality samples",
                        int((predictions == 1).sum()),
                    )
                    st.download_button(
                        "Download prediction results",
                        results.to_csv(index=False),
                        "wine_quality_predictions.csv",
                        "text/csv",
                    )

st.info(
    "Responsible use: this is a classroom prototype based on public red-wine data. "
    "A prediction is a preliminary screening aid, not a substitute for laboratory "
    "testing, sensory evaluation, food-safety procedures, or professional "
    "quality-control decisions."
)
