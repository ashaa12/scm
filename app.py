import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("random_forest_model.joblib")

st.title("Prediksi Weekday Order - Random Forest")

# input form
uploaded_file = st.file_uploader("Upload CSV untuk prediksi", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # encode categorical
    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        df[col] = df[col].astype("category").cat.codes

    pred = model.predict(df)
    df["Prediksi_Weekday_Order"] = pred

    st.write("Hasil Prediksi:")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Hasil Prediksi", csv, "prediksi.csv")
