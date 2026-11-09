# Wine Quality Prediction System

A classroom prototype for the Data Mining Laboratory (Section B). It uses physicochemical measurements to classify red-wine samples as **GOOD QUALITY** or **NEEDS IMPROVEMENT** and provides a CSV upload/download workflow in Streamlit.

## Problem and SAD mini-analysis

- **Problem:** A producer needs a consistent preliminary way to screen wine samples from recorded physicochemical measurements.
- **Users:** Quality-control staff, production supervisors, or data analysts.
- **Inputs:** Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free and total sulfur dioxide, density, pH, sulphates, and alcohol.
- **Process:** Validate the CSV columns and numeric measurements, apply the trained preprocessing/model pipeline, and classify each sample.
- **Outputs:** A predicted category, downloadable results CSV, and a held-out evaluation summary (accuracy, per-class precision/recall/F1, and confusion matrix).
- **Key limitation:** The model uses a particular red-wine dataset and may not generalize to other wine types, producers, regions, or testing conditions.

## Dataset and prediction rule

Dataset: [UCI Machine Learning Repository — Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), red-wine CSV.

- `quality >= 6` → **GOOD QUALITY** (class 1)
- `quality < 6` → **NEEDS IMPROVEMENT** (class 0)

The original `quality` column is the target used during training only. It is excluded from predictor features to prevent target leakage.

## Technologies

Python 3.10+, pandas, scikit-learn (Random Forest and preprocessing pipeline), joblib, Streamlit, and GitHub.

## Installation and run

Run these commands from this directory:

```bash
pip install -r requirements.txt
python train_model.py      # downloads data, evaluates model, saves wine_quality.csv and wine_model.pkl
streamlit run app.py       # opens the local app, normally at http://localhost:8501
```

The app downloads the UCI CSV if it is not present and trains/saves the model if `wine_model.pkl` is missing. It displays the dataset preview and a reproducible stratified 80/20 evaluation. Download the sample input CSV, upload it (or another CSV with all required feature columns), review predictions, and download the results. Blank numeric values are imputed by the model pipeline; missing columns and invalid non-numeric/infinite values are reported.

## Project files

- `train_model.py` — downloads and inspects the data, removes duplicate training rows, trains a Random Forest pipeline, prints evaluation metrics, and saves the model.
- `app.py` — Streamlit dataset dashboard, evaluation summary, CSV validation and prediction, and result export.
- `requirements.txt` — Python package dependencies.
- `wine_quality.csv` — downloaded red-wine dataset (if included locally/repository).
- `wine_model.pkl` — generated model artifact; regenerate with `python train_model.py` if needed.

## Testing checklist

Run each test locally and record the actual result for submission:

| Test | Expected result |
| --- | --- |
| `python train_model.py` | Dataset and model files are created; metrics print without errors. |
| `streamlit run app.py` | Dashboard, dataset preview, and evaluation summary load. |
| Upload the downloaded sample CSV | Predictions display and results can be downloaded as CSV. |
| Upload a CSV missing a predictor column | The app lists the missing required column(s). |
| Upload malformed/empty CSV or invalid measurement values | The app gives a helpful error; blank measurements are handled by pipeline imputation. |

## Reflection prompts

1. **What is the target, and why is `quality` excluded from inputs?** The binary target is `quality >= 6`; the original score is excluded because it directly defines the target and including it would leak the answer to the model.
2. **What does GOOD QUALITY recall mean?** It is the fraction of actually good samples that the model identifies as good. Low recall means more good samples may be missed during screening.
3. **Why can performance differ on another producer's wine?** Producers, grape varieties, regions, measurement equipment, and production conditions can shift the data distribution from the training set.
4. **Why is this preliminary only?** The data and model are limited, and a model prediction cannot replace laboratory tests, sensory evaluation, food-safety procedures, or expert quality-control decisions.
5. **What could improve reliability?** More representative and independently collected data, validated measurements, samples across producers/regions/vintages, external testing, repeated validation, and review of class-specific errors.

## Responsible use and limitations

This is a classroom prototype based on a public dataset. Treat predictions as an initial screening aid only—not as a final quality decision or a substitute for laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions. Accuracy alone is not sufficient; review per-class precision, recall, F1-score, and the confusion matrix.
