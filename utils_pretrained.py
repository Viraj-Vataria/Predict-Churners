import json
import pandas as pd
import joblib

MODEL_PATH = "pretrained_model.pkl"
SCHEMA_PATH = "feature_schema.json"

model = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH) as f:
    REQUIRED_COLS = json.load(f)["required_columns"]


def validate_schema(df):
    missing = set(REQUIRED_COLS) - set(df.columns)
    extra = set(df.columns) - set(REQUIRED_COLS)

    return missing, extra


def predict_pretrained(df):
    df = df[REQUIRED_COLS]
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    df["Predicted_Churn"] = preds
    df["Churn_Probability"] = probs

    return df
