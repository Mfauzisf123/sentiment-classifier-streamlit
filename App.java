import streamlit as st
import pandas as pd
import joblib

# Load model dan vectorizer
model = joblib.load("model_sentimen.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Klasifikasi Ulasan Sentimen")
st.write("Masukkan teks ulasan produk dan sistem akan mengklasifikasikan sentimennya (positif/negatif).")

# Input teks dari pengguna
input_text = st.text_area("Masukkan ulasan:", "")

if st.button("Klasifikasikan"):
    if input_text.strip() == "":
        st.warning("Mohon masukkan teks ulasan terlebih dahulu.")
    else:
        # Preprocessing (bisa disesuaikan jika ada di notebook)
        transformed = vectorizer.transform([input_text])
        prediction = model.predict(transformed)[0]
        label = "Positif" if prediction == 1 else "Negatif"
        st.success(f"Prediksi Sentimen: **{label}**")
