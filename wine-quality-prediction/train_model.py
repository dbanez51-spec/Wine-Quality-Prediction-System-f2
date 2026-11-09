# train_model.py - Wine Quality Prediction System (Section B)
# Run with: python train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------- Activity A: Retrieve and inspect the dataset ----------------

# Download the red-wine CSV from the UCI repository (columns are separated by ";")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")

# Save a local copy (the Streamlit app reads this file)
df.to_csv("wine_quality.csv", index=False)

# Inspect the data
print("Rows and columns:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Quality distribution:")
print(df["quality"].value_counts().sort_index())

# ---------------- Activity B: Preprocess, train, and evaluate ----------------

# Remove duplicate records
df = df.drop_duplicates().copy()

# Target: 1 = Good quality (quality >= 6), 0 = Needs improvement (quality < 6)
y = (df["quality"] >= 6).astype(int)

# Inputs: every column EXCEPT quality (keeping quality would be target leakage)
X = df.drop(columns=["quality"])

# Split 80% train / 20% test, keeping the same class proportions in both
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# One pipeline: fill missing values -> scale numbers -> Random Forest
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    ))
])

# Train the model
model.fit(X_train, y_train)

# Test the model on data it has never seen
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))
print(classification_report(
    y_test, predictions,
    target_names=["Needs improvement", "Good quality"],
    zero_division=0
))

# Save the trained model for the Streamlit app
joblib.dump(model, "wine_model.pkl")
print("Saved wine_model.pkl")
