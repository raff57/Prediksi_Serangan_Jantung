import streamlit as st
import pandas as pd
import joblib

# Load pipeline model (pastikan file .pkl-nya ada)
model = joblib.load('pipe_knn.pkl')

def run():
    st.title("Ayo Prediksi Apakah kamu Salah Satu Yang Memiliki Resiko Serangan Jantung")

    st.markdown("Masukkan data Kamu di bawah ini untuk memprediksi risiko serangan jantung:")

    # Form input
    with st.form("input_form"):
        age = st.number_input("Umur", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        diet = st.selectbox("Jenis Pola Makan", ["Healthy", "Unhealthy"])
        cholesterol = st.number_input("Kolesterol", min_value=100, max_value=400, value=200)
        heart_rate = st.number_input("Detak Jantung", min_value=40, max_value=200, value=80)
        diabetes = st.selectbox("Punya Diabetes?", ["Tidak", "Ya"])
        family_history = st.selectbox("Riwayat Keluarga?", ["Tidak", "Ya"])
        smoking = st.selectbox("Merokok?", ["Tidak", "Ya"])
        obesity = st.selectbox("Obesitas?", ["Tidak", "Ya"])
        alcohol_consumption = st.slider("Konsumsi Alkohol (0=none, 4=tinggi)", 0, 4, 1)
        exercise_hours = st.slider("Jam Olahraga per Minggu", 0, 20, 2)
        previous_heart_problems = st.selectbox("Pernah Masalah Jantung?", ["Tidak", "Ya"])
        medication_use = st.selectbox("Pakai Obat?", ["Tidak", "Ya"])
        stress_level = st.slider("Tingkat Stres (0–10)", 0, 10, 5)
        sedentary_hours = st.slider("Jam Duduk per Hari", 0, 20, 8)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
        triglycerides = st.number_input("Trigliserida", min_value=50, max_value=500, value=150)
        physical_activity_days = st.slider("Hari Aktif Fisik/Minggu", 0, 7, 3)
        sleep_hours = st.slider("Jam Tidur per Hari", 0, 24, 7)
        systolic = st.number_input("Tekanan Darah Sistolik", min_value=80, max_value=200, value=120)
        diastolic = st.number_input("Tekanan Darah Diastolik", min_value=50, max_value=140, value=80)

        submitted = st.form_submit_button("Prediksi")
        data = {
            'age': age,
            'cholesterol': cholesterol,
            'heart_rate': heart_rate,
            'diabetes': 1 if diabetes == "Ya" else 0,
            'family_history': 1 if family_history == "Ya" else 0,
            'smoking': 1 if smoking == "Ya" else 0,
            'obesity': 1 if obesity == "Ya" else 0,
            'alcohol_consumption': alcohol_consumption,
            'exercise_hours_per_week': exercise_hours,
            'previous_heart_problems': 1 if previous_heart_problems == "Ya" else 0,
            'medication_use': 1 if medication_use == "Ya" else 0,
            'stress_level': stress_level,
            'sedentary_hours_per_day': sedentary_hours,
            'bmi': bmi,
            'triglycerides': triglycerides,
            'physical_activity_days_per_week': physical_activity_days,
            'sleep_hours_per_day': sleep_hours,
            'systolic': systolic,
            'diastolic': diastolic,
            'sex_Male': 1 if sex == "Male" else 0,
            'diet_Healthy': 1 if diet == "Healthy" else 0,
            'diet_Unhealthy': 1 if diet == "Unhealthy" else 0,
        }
    if submitted:


        df = pd.DataFrame([data])
        pred = model.predict(df)[0]
        label = "Risiko Tinggi" if pred == 1 else " Risiko Rendah"
        st.subheader("Hasil Prediksi:")
        st.success(label)
