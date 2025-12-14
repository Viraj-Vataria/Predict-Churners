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
    
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.info(f"**Total rows:** {len(df)}")

    # ===== MODE 1: PRETRAINED MODEL =====
    if mode == "Pretrained Churn Model":
        missing, extra = validate_schema(df)
        
        if missing:
            st.error(f"❌ Missing required columns: {', '.join(missing)}")
            st.stop()
        
        if extra:
            st.warning(f"⚠️ Extra columns will be ignored: {', '.join(extra)}")
        
        if st.button("🎯 Predict Churn", type="primary"):
            with st.spinner("Running predictions..."):
                result_df = predict_pretrained(df)
            
            st.success("✅ Prediction completed!")
            
            # Show results
            st.subheader("📈 Prediction Results")
            st.dataframe(result_df, use_container_width=True)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            churn_count = (result_df["Churn_Prediction"] == "Yes").sum()
            churn_pct = (churn_count / len(result_df)) * 100
            
            with col1:
                st.metric("Total Customers", len(result_df))
            with col2:
                st.metric("Predicted Churners", churn_count)
            with col3:
                st.metric("Churn Rate", f"{churn_pct:.1f}%")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(6, 4))
            churn_counts = result_df["Churn_Prediction"].value_counts()
            colors = ["#2ecc71", "#e74c3c"]
            ax.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
            ax.set_title("Churn Distribution")
            st.pyplot(fig)
            
            # Download button
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

    # ===== MODE 2: DYNAMIC TRAINING =====
    else:
        target = st.selectbox("Select Target Column (Churn)", df.columns)
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                model, acc = train_dynamic_model(df, target)
            
            st.metric("Model Accuracy", f"{acc:.2%}")
            
            preds = model.predict(df.drop(columns=[target]))
            df["Predicted_Churn"] = pd.Series(preds).map({0: "No", 1: "Yes"})
            
            st.dataframe(df.head(), use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="dynamic_predictions.csv",
                mime="text/csv"
            )
