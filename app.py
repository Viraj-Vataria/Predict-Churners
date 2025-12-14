import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils_dynamic import train_dynamic_model
from utils_pretrained import validate_schema, predict_pretrained

st.set_page_config("Churn Prediction Platform", layout="wide")

st.title("🔮 Churn Prediction Platform")
st.write("Choose between a pretrained enterprise model or dynamic model training.")

mode = st.radio(
    "Select Prediction Mode",
    ["Pretrained Churn Model", "Train Model on Uploaded Dataset"]
)

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ===== MODE 1 =====
    if mode == "Pretrained Churn Model":
        missing, extra = validate_schema(df)

        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        if extra:
            st.warning(f"Extra columns will be ignored: {extra}")

        if st.button("Predict Churn"):
            result_df = predict_pretrained(df)

            st.success("Prediction completed")
            st.dataframe(result_df.head())

            fig, ax = plt.subplots()
            sns.countplot(x=result_df["Predicted_Churn"], ax=ax)
            st.pyplot(fig)

            st.download_button(
                "Download Predictions",
                result_df.to_csv(index=False),
                "pretrained_predictions.csv",
                "text/csv"
            )

    # ===== MODE 2 =====
    else:
        target = st.selectbox("Select Target Column", df.columns)

        if st.button("Train Model"):
            with st.spinner("Training model..."):
                model, acc = train_dynamic_model(df, target)

            st.metric("Model Accuracy", f"{acc:.2f}")

            preds = model.predict(df.drop(columns=[target]))
            df["Predicted_Churn"] = preds

            st.dataframe(df.head())

            st.download_button(
                "Download Predictions",
                df.to_csv(index=False),
                "dynamic_predictions.csv",
                "text/csv"
            )
