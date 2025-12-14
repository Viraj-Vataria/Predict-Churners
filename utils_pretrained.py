import pandas as pd
import numpy as np
import joblib

MODEL_PATH = "viraj_model.pkl"
model = joblib.load(MODEL_PATH)

def _get_expected_columns(m):
    # Works for many sklearn estimators + Pipelines
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    if hasattr(m, "named_steps"):
        # If pipeline exposes feature_names_in_ at pipeline level, above already handled.
        # Fall back to common step names.
        for step_name in ["preprocessor", "preprocess", "columntransformer"]:
            if step_name in m.named_steps and hasattr(m.named_steps[step_name], "feature_names_in_"):
                return list(m.named_steps[step_name].feature_names_in_)
    return None

EXPECTED_COLS = _get_expected_columns(model)

def prepare_input(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # Drop columns that should not go into features
    for c in ["Churn", "churn", "customerID", "CustomerID"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # If model tells us what it expects, enforce it
    if EXPECTED_COLS is not None:
        # Add missing expected cols as NaN (pipeline/imputer can handle if it exists)
        missing = [c for c in EXPECTED_COLS if c not in df.columns]
        for c in missing:
            df[c] = np.nan

        # Keep only expected columns and correct order
        df = df[EXPECTED_COLS]

    # Type cleaning that commonly breaks churn datasets
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df

def predict_pretrained(raw_df: pd.DataFrame) -> pd.DataFrame:
    X = prepare_input(raw_df)

    preds = model.predict(X)

    # Decide which label means "churn" using model.classes_ when available
    pos_label = None
    if hasattr(model, "classes_") and len(getattr(model, "classes_", [])) == 2:
        classes = list(model.classes_)
        # Prefer common "positive" labels if present; otherwise use the 2nd class
        for candidate in [1, "Yes", "YES", True, "True", "Churn", "churn"]:
            if candidate in classes:
                pos_label = candidate
                break
        if pos_label is None:
            pos_label = classes[1]  # fallback

    # Probability of the positive class (if supported)
    prob = None
    if hasattr(model, "predict_proba") and pos_label is not None and hasattr(model, "classes_"):
        classes = list(model.classes_)
        pos_index = classes.index(pos_label)
        prob = model.predict_proba(X)[:, pos_index]

    out = raw_df.copy()

    # Force Yes/No output no matter what preds look like
    if pos_label is not None:
        out["Churn_Prediction"] = np.where(pd.Series(preds) == pos_label, "Yes", "No")
    else:
        # If we cannot infer pos_label, fall back to safe string conversion
        out["Churn_Prediction"] = pd.Series(preds).astype(str)

    if prob is not None:
        out["Churn_Probability"] = prob

    # Move prediction columns to the end
    for col in ["Churn_Prediction", "Churn_Probability"]:
        if col in out.columns:
            tmp = out.pop(col)
            out[col] = tmp

    return out
