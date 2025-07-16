import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from PIL import Image

def run():
    st.write('# RESIKO TERKENA SERANGAN JANTUNG')

    # load
    gambar = Image.open('serangan_jantung.jpg')
    st.write(gambar)

    st.write('# Penjelasan')

    st.write('''Serangan jantung, atau dalam istilah medis disebut infark miokard, 
             adalah kondisi serius yang terjadi ketika aliran darah ke bagian otot 
             jantung terhambat atau terhenti sama sekali. Hal ini biasanya disebabkan 
             oleh penyumbatan pada pembuluh darah koroner akibat penumpukan plak 
             kolesterol (aterosklerosis) yang pecah dan membentuk gumpalan darah. 
             Ketika suplai oksigen ke jantung terganggu, sel-sel jantung mulai mati, 
             yang dapat menyebabkan kerusakan permanen pada organ tersebut. Serangan 
             jantung sering kali ditandai dengan nyeri dada, sesak napas, mual, 
             keringat dingin, dan rasa tidak nyaman pada lengan, leher, atau rahang. 
             Faktor risiko utama meliputi tekanan darah tinggi, kolesterol tinggi, 
             diabetes, kebiasaan merokok, obesitas, stres, dan gaya hidup yang kurang 
             aktif. Pencegahan dini melalui pola hidup sehat, deteksi risiko secara 
             rutin, dan pengelolaan stres dapat secara signifikan menurunkan kemungkinan terjadinya serangan jantung.''')

    # load data
    df = pd.read_csv('heart_attack_prediction_dataset.csv')

    st.write('# Sumber Data')
    # menampilkan dataframe
    st.write(df)

    # membuat visualisasi (grafik)
    st.write('# Explorasi Analisis Data')

    st.write('## Distribubusi dari Resiko Serangan Jantung')

    # EDA 1 - Distribusi Risiko Serangan Jantung
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='Heart Attack Risk', ax=ax)
    ax.set_title('Distribusi Risiko Serangan Jantung')
    ax.set_xlabel('Heart Attack Risk')
    ax.set_ylabel('Jumlah Pasien')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Tidak Risiko', 'Berisiko'])
    st.pyplot(fig)
    st.write('''Grafik di atas menunjukkan distribusi kelas target heart_attack_risk, yang mengindikasikan apakah seorang pasien berisiko terkena serangan jantung atau tidak. Terlihat bahwa jumlah pasien yang tidak berisiko (label 0) jauh lebih banyak dibandingkan dengan pasien yang berisiko (label 1). Hal ini menunjukkan adanya ketidakseimbangan kelas (class imbalance) dalam dataset.

Dengan kata lain, mayoritas data berada pada kelas “tidak berisiko”, sementara jumlah data untuk kelas “berisiko” relatif lebih sedikit. Kondisi seperti ini penting untuk diperhatikan karena dapat memengaruhi kinerja model machine learning — model cenderung bias terhadap kelas mayoritas dan dapat mengabaikan kasus-kasus minoritas yang justru penting, seperti pasien yang benar-benar berisiko. 
            ''')
    
    # EDA 2
    st.subheader("Distribusi Usia Pasien")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df, x='Age', kde=True, bins=30) 
    ax.set_title('Distribusi Usia Pasien')
    ax.set_xlabel('Usia')
    ax.set_ylabel('Jumlah Pasien')
    st.pyplot(fig)
    st.write('''Grafik ini menunjukkan distribusi usia pasien yang menjadi bagian dari dataset. Dari visualisasi histogram, terlihat bahwa data pasien tersebar secara relatif merata di rentang usia 20 hingga 90 tahun. Tidak terdapat puncak distribusi yang menonjol pada kelompok usia tertentu, yang berarti pasien dalam dataset ini datang dari berbagai rentang usia tanpa dominasi yang signifikan.

Garis KDE (Kernel Density Estimation) menunjukkan kecenderungan jumlah pasien di setiap kelompok usia, dengan sedikit fluktuasi tetapi tetap menunjukkan distribusi yang cukup seimbang. Hal ini menunjukkan bahwa tidak ada bias usia dalam pengambilan data pasien, dan model dapat mempelajari risiko serangan jantung dari populasi usia yang beragam.
            ''')

    # EDA 3
    st.subheader("Distribusi Usia berdasarkan Risiko Serangan Jantung")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Heart Attack Risk', y='Age', palette='Set2', ax=ax)
    ax.set_title('Distribusi Usia berdasarkan Risiko Serangan Jantung')
    ax.set_xlabel('Heart Attack Risk')
    ax.set_ylabel('Usia')
    ax.set_xticklabels(['Tidak Risiko', 'Berisiko'])  # 0 → Tidak Risiko, 1 → Berisiko
    st.pyplot(fig)
    st.write('''Grafik ini memperlihatkan distribusi usia pasien yang dikelompokkan berdasarkan status risiko serangan jantung. Dari visualisasi boxplot terlihat bahwa baik kelompok pasien yang tidak berisiko maupun yang berisiko memiliki rentang usia yang hampir sama, yaitu sekitar 18 hingga 90 tahun. Median usia untuk kedua kelompok pun relatif berdekatan, yaitu sekitar usia 55 tahun.

Artinya, usia bukan satu-satunya faktor dominan dalam membedakan kelompok risiko. Baik pasien muda maupun lanjut usia sama-sama bisa berada di kelompok berisiko, tergantung pada faktor-faktor lainnya seperti gaya hidup, riwayat kesehatan, dan kondisi medis lain. Grafik ini menunjukkan bahwa distribusi usia di antara pasien yang berisiko dan tidak berisiko cukup seimbang, sehingga analisis prediktif perlu memperhatikan kombinasi variabel lain, bukan hanya usia semata.
            ''')
    
    # EDA 4
    st.subheader("Risiko Serangan Jantung Berdasarkan Jenis Kelamin")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='Sex', hue='Heart Attack Risk', palette='pastel', ax=ax)
    ax.set_title('Risiko Serangan Jantung Berdasarkan Jenis Kelamin')
    ax.set_xlabel('Jenis Kelamin')
    ax.set_ylabel('Jumlah Pasien')
    ax.legend(title='Heart Attack Risk', labels=['Tidak Risiko', 'Berisiko'])
    st.pyplot(fig)
    st.write('''Grafik ini menampilkan perbandingan jumlah pasien berdasarkan jenis kelamin dan status risiko serangan jantung. Terlihat bahwa laki-laki (Male) memiliki jumlah pasien yang lebih tinggi dibandingkan perempuan (Female), baik pada kategori tidak berisiko maupun berisiko.

Namun, jika dilihat secara proporsional, jumlah pasien laki-laki yang berisiko mengalami serangan jantung lebih tinggi dibandingkan perempuan. Artinya, jenis kelamin laki-laki cenderung lebih rentan terhadap risiko serangan jantung dalam dataset ini. Hal ini sejalan dengan beberapa penelitian medis yang menyatakan bahwa pria memiliki kemungkinan lebih besar terkena penyakit jantung di usia lebih muda dibandingkan wanita, yang biasanya terlindungi oleh faktor hormonal sebelum menopause.

Dengan demikian, faktor jenis kelamin dapat menjadi salah satu indikator penting dalam analisis risiko serangan jantung.
            ''')

    # EDA 5
    st.subheader("Distribusi BMI berdasarkan Risiko Serangan Jantung")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Heart Attack Risk', y='BMI', palette='coolwarm', ax=ax)
    ax.set_title('Distribusi BMI berdasarkan Risiko Serangan Jantung')
    ax.set_xlabel('Heart Attack Risk')
    ax.set_ylabel('BMI')
    ax.set_xticklabels(['Tidak Risiko', 'Berisiko'])
    st.pyplot(fig)
    st.write('''Grafik ini menunjukkan distribusi nilai Body Mass Index (BMI) pada dua kelompok pasien, yaitu yang tidak berisiko (label 0) dan yang berisiko (label 1) mengalami serangan jantung. Secara visual, terlihat bahwa sebaran nilai BMI antara kedua kelompok cukup mirip, dengan rentang nilai yang hampir sama.

Median BMI kedua kelompok juga hampir identik, yang menunjukkan bahwa rata-rata indeks massa tubuh tidak terlalu berbeda antara pasien yang berisiko dan tidak berisiko. Namun, nilai kuartil atas pada kelompok berisiko sedikit lebih tinggi, yang bisa mengindikasikan bahwa beberapa pasien berisiko memiliki BMI lebih tinggi dari rata-rata.

Hal ini bisa diartikan bahwa meskipun BMI bukan satu-satunya faktor penentu risiko serangan jantung, pasien dengan nilai BMI tinggi tetap memiliki kecenderungan lebih tinggi terhadap risiko tersebut. Oleh karena itu, pengelolaan berat badan tetap penting sebagai bagian dari pencegahan penyakit jantung.
            ''')

    # EDA 6
    st.subheader("Distribusi Kolesterol berdasarkan Risiko Serangan Jantung")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Heart Attack Risk', y='Cholesterol', palette='viridis', ax=ax)
    ax.set_title('Distribusi Kolesterol berdasarkan Risiko Serangan Jantung')
    ax.set_xlabel('Heart Attack Risk')
    ax.set_ylabel('Kolesterol')
    ax.set_xticklabels(['Tidak Risiko', 'Berisiko'])
    st.pyplot(fig)
    st.write('''Grafik di atas memperlihatkan distribusi kadar kolesterol pada dua kelompok pasien, yaitu pasien yang tidak berisiko (label 0) dan yang berisiko (label 1) mengalami serangan jantung. Dari visualisasi boxplot ini, terlihat bahwa distribusi kolesterol antara kedua kelompok hampir serupa, baik dari sisi rentang (range) maupun nilai tengah (median).

Nilai median kolesterol untuk kedua kelompok berada pada kisaran yang hampir sama, yang mengindikasikan bahwa rata-rata kadar kolesterol tidak jauh berbeda antara pasien berisiko dan tidak berisiko. Namun, terdapat penyebaran data yang cukup luas di kedua kelompok, dengan beberapa outlier di area kadar kolesterol tinggi.

Hal ini menunjukkan bahwa kolesterol tetap menjadi faktor penting yang perlu dikontrol, meskipun secara distribusi umum tidak menunjukkan perbedaan mencolok antara dua kelompok risiko. Pemantauan kadar kolesterol secara rutin sangat dianjurkan sebagai bagian dari langkah preventif terhadap penyakit jantung.
            ''')
    
    # EDA 7
    st.subheader("Distribusi Tingkat Stres berdasarkan Risiko Serangan Jantung")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Heart Attack Risk', y='Stress Level', palette='flare', ax=ax)
    ax.set_title('Distribusi Tingkat Stres berdasarkan Risiko Serangan Jantung')
    ax.set_xlabel('Heart Attack Risk')
    ax.set_ylabel('Tingkat Stres')
    ax.set_xticklabels(['Tidak Risiko', 'Berisiko'])
    st.pyplot(fig)
    st.write('''Grafik boxplot ini menampilkan distribusi tingkat stres pada dua kelompok pasien, yaitu yang tidak berisiko (label 0) dan yang berisiko (label 1) terkena serangan jantung. Dari visualisasi tersebut, terlihat bahwa distribusi tingkat stres pada kedua kelompok cukup mirip. Nilai median tingkat stres hampir sama, berada di sekitar nilai 5, dengan rentang distribusi yang juga serupa.

Artinya, secara umum, tingkat stres dialami oleh pasien dalam kedua kategori risiko tidak menunjukkan perbedaan yang signifikan. Meski demikian, hal ini tidak serta-merta meniadakan peran stres sebagai faktor risiko, karena stres tetap dapat berkontribusi secara tidak langsung terhadap kesehatan jantung jika dikombinasikan dengan faktor lain seperti tekanan darah, pola tidur, dan gaya hidup.

Oleh karena itu, penting untuk tetap memperhatikan dan mengelola tingkat stres sebagai bagian dari upaya menjaga kesehatan jantung, meskipun dari grafik ini stres tampaknya tidak menjadi faktor pembeda yang dominan antara kelompok berisiko dan tidak berisiko.
            ''')

if __name__ == '__main__':
    run()
