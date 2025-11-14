from my_transformers import MyTransformer
import streamlit as st
import joblib
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt


model = joblib.load("rf_model_compressed.pkl")

st.title("Batch Customer Churn Prediction App")
st.write("Upload a CSV file with customer data to predict if they will stay or leave the company.")


uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)
    st.write("File uploaded successfully. Here’s a preview:")
    st.dataframe(df.head())

    
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    
    required_columns = [
        "age", "tenure", "usage_frequency", "support_calls",
        "payment_delay", "subscription_type", "contract_length",
        "total_spend", "last_interaction"
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        if st.button("Predict for all customers"):
            
            preds = model.predict(df)
            df["prediction"] = ["WILL LEAVE" if p == 1 else "WILL STAY" for p in preds]

            st.success("Predictions completed successfully!")
            st.dataframe(df.head())

            st.subheader("Prediction Summary")
            summary = df["prediction"].value_counts()

            fig, ax = plt.subplots()
            ax.pie(
                summary,
                labels=summary.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=["#4CAF50","#FF6B6B"]
            )
            ax.axis("equal")  
            st.pyplot(fig)

            output = BytesIO()
            df.to_csv(output, index=False)
            processed_data = output.getvalue()

            st.download_button(
                label="Download Predicted CSV",
                data=processed_data,
                file_name="predicted_customers.csv",
                mime="text/csv"
            )
