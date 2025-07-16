import eda, prediction
import streamlit as st

with st.sidebar:
    st.write('# HALAMAN')

    page = st.selectbox('Pilih Halaman',['BERANDA','PREDIKSI DIRI KAMU'])

    st.write ('# Tentang')
    st.markdown ('Aplikasi ini tidak dimaksudkan sebagai pengganti diagnosis medis, tetapi sebagai alat bantu untuk meningkatkan kesadaran terhadap kesehatan jantung dan pentingnya deteksi dini. Diharapkan dengan adanya alat ini, masyarakat dapat lebih proaktif dalam menjaga gaya hidup sehat dan berkonsultasi lebih dini dengan tenaga medis profesional.')

if page == 'BERANDA':
    eda.run()

if page == 'PREDIKSI DIRI KAMU':
    prediction.run()
