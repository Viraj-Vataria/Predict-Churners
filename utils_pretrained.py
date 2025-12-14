import json
import pandas as pd
import joblib

MODEL_PATH = "viraj_model.pkl"
SCHEMA_PATH = "feature_schema.json"

model = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH) as f:
    REQUIRED_COLS = json.load(f)["required_columns"]

def validate_schema(df):
    missing = set(REQUIRED_COLS) - set(df.columns)
    extra = set(df.columns) - set(REQUIRED_COLS) - {"customerID"}
    return missing, extra

def predict_pretrained(df):
    # Keep original dataframe structure
    result_df = df.copy()
    
    # Extract only model features for prediction
    X = df[REQUIRED_COLS].copy()
    
    # Clean TotalCharges if needed
    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)
    
    # Get predictions
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    
    # Map 0/1 to No/Yes
    result_df["Churn_Prediction"] = pd.Series(preds).map({0: "No", 1: "Yes"})
    result_df["Churn_Probability"] = probs
    
    # Move prediction columns to the end
    for col in ["Churn_Prediction", "Churn_Probability"]:
        temp = result_df.pop(col)
        result_df[col] = temp
    
    return result_df
