# Wine Quality Prediction System

A classroom prototype (Data Mining Laboratory, Section B) that classifies a red-wine sample as **GOOD QUALITY** or **NEEDS IMPROVEMENT** from its physicochemical measurements.

## Problem
A small wine producer needs a consistent, preliminary way to screen wine samples using recorded measurements before further quality assessment.

## Dataset
UCI Machine Learning Repository - Wine Quality (red wine): https://archive.ics.uci.edu/dataset/186/wine+quality

## Prediction rule
- `quality >= 6` -> GOOD QUALITY
- `quality < 6` -> NEEDS IMPROVEMENT

The original `quality` score is the target only and is never used as an input feature.

## Technologies
Python 3.10+, pandas, scikit-learn (Random Forest), joblib, Streamlit, GitHub

## Installation and run
```bash
pip install -r requirements.txt
python train_model.py      # downloads data, trains model, creates wine_quality.csv and wine_model.pkl
streamlit run app.py       # opens the app at http://localhost:8501
```
In the app, download the sample input CSV and upload it to see predictions.

## Limitations
- Trained on one red-wine dataset; it may not work well for other wine types, producers, regions, or testing conditions.
- A prediction is a preliminary screening aid, not a substitute for laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.
