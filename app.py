import streamlit as st
import pandas as pd
from utils_pretrained import predict_pretrained

st.set_page_config(page_title="Churn Prediction Platform", layout="wide")

st.title("Churn Prediction Platform")
st.write("Upload a dataset and download the same file with churn predictions added.")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    if st.button("Predict Churn", type="primary"):
        with st.spinner("Predicting..."):
            result_df = predict_pretrained(df)

        st.success("Done! Predictions added as the last column.")
        st.dataframe(result_df.head(50), use_container_width=True)

        st.download_button(
            "Download file with predictions",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv",
        )
