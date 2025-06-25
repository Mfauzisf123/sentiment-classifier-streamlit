import streamlit as st
import joblib
import os

st.set_page_config(page_title="Klasifikasi Ulasan Sentimen", layout="centered")

st.title("📊 Klasifikasi Ulasan Sentimen")
st.write("Masukkan teks ulasan produk dan sistem akan mengklasifikasikan sentimennya (positif atau negatif).")

# Coba load model & vectorizer
model_path = "model_sentimen.pkl"
vec_path = "vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vec_path):
    st.error("❌ File model atau vectorizer tidak ditemukan. Pastikan kedua file `.pkl` sudah diupload.")
else:
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)

        input_text = st.text_area("📝 Masukkan ulasan:", "", height=150)

        if st.button("🔍 Klasifikasikan"):
            if input_text.strip() == "":
                st.warning("⚠️ Mohon masukkan teks ulasan terlebih dahulu.")
            else:
                # Lakukan transformasi teks
                transformed = vectorizer.transform([input_text])
                prediction = model.predict(transformed)[0]
                label = "🟢 Positif" if prediction == 1 else "🔴 Negatif"
                st.success(f"Hasil Prediksi Sentimen: **{label}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat load model atau prediksi: {str(e)}")
