import streamlit as st
import pandas as pd
import joblib

# ====== 1. Load model & encoders ======
model = joblib.load("model_paru.pkl")
encoders = joblib.load("encoders_paru.pkl")
feature_cols = list(encoders.keys())

# ====== 2. Fungsi prediksi aman ======
def predict_safe(input_dict):
    df_in = pd.DataFrame([input_dict], columns=feature_cols)
    for col in feature_cols:
        val = str(df_in.loc[0, col])
        le = encoders[col]
        if val not in le.classes_:
            raise ValueError(
                f"Kategori '{val}' tidak valid untuk '{col}'. Pilih dari {list(le.classes_)}"
            )
        df_in[col] = le.transform([val])
    pred = model.predict(df_in)
    return int(pred[0])

# ====== 3. UI Streamlit ======
st.title("🫁 Prediksi Penyakit Paru-Paru")
st.write("Aplikasi ini memprediksi apakah pasien terkena penyakit paru-paru atau tidak.")

# ====== 4. Contoh pasien default ======
# Silakan sesuaikan dengan dataset kamu
default_input = {
    "Usia": "Tua",
    "Jenis_Kelamin": "Laki-laki",
    "Merokok": "Ya",
    "Batuk": "Ya",
    "Sesak_Napas": "Ya",
    "Riwayat_Keluarga": "Ya",
    "Asuransi": "Tidak"
}

# Kalau ada kolom lain di dataset, tambahkan di dictionary default_input

# ====== 5. Input user via selectbox ======
user_input = {}
for col in feature_cols:
    choices = list(encoders[col].classes_)
    if col in default_input and default_input[col] in choices:
        default_value = default_input[col]
    else:
        default_value = choices[0]  # fallback
    user_input[col] = st.selectbox(col, choices, index=choices.index(default_value))

# ====== 6. Tombol Prediksi ======
if st.button("Prediksi"):
    try:
        result = predict_safe(user_input)
        if result == 1:
            st.error("⚠️ Pasien diprediksi TERKENA penyakit paru-paru")
        else:
            st.success("✅ Pasien diprediksi TIDAK terkena penyakit paru-paru")
    except Exception as e:
        st.error(str(e))
