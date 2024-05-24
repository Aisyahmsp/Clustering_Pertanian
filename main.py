import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import altair as alt
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

st.markdown("""<h1 style='text-align: center;'> Pengelompokan Produktivitas Pertanian </h1> """, unsafe_allow_html=True)
# 1. as sidevar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Pilihan Menu", #required
        options=["Beranda", "Deskripsi", "Dataset", "Preprocessing", "Clustering", "Evaluasi", "Referensi"], #required
        icons=["house-door-fill", "book-half", "bi bi-file-earmark-arrow-up-fill", "arrow-repeat","medium", "folder-fill", "bookmark-fill"], #optional
        menu_icon="cast", #optional
        default_index=0, #optional    
    styles={
        "container": {"padding": "0!important", "background-color":"white"},
        "icon": {"color": "black", "font-size": "17px"},
        "nav-link": {
            "font-size": "17px",
            "text-align": "left",
            "margin": "0px",
            "--hover-color": "#4169E1",
        },
        "nav-link-selected": {"background-color": "Royalblue"}
    }
    )


if selected == "Beranda":
    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        st.write("")

    with col2:
        img = Image.open('pertanian.jpeg')
        st.image(img, use_column_width=False, width=300)

    with col3:
        st.write("")

    st.write(""" """)
    
    
    st.write(""" """)

    st.write("""
    Pengelolaan pertanian yang efektif dengan memahami pola produktivitas pertanian di berbagai kelompok di Sumenep sangat diperlukan.
    Hal tersebut dikarenakan produktivitas pertanian di Sumenep sangat beragam jenisnya, 
    sehingga potensi hasil produksi di kelompok satu dengan kelompok lainnya belum tentu sama. 
    Dengan melakukan analisis pola produktivitas, para petani, pemerintah, dan pemangku kebijakan lainnya 
    dapat terbantu untuk melakukan berbagai pengambilan keputusan yang lebih baik terkait alokasi sumber daya, 
    pengembangan pertanian berkelanjutan, dan pemberdayaan ekonomi desa.

    Salah satu pendekatan yang umum digunakan dalam menganalisis pola produktivitas adalah teknik clustering (pengelompokan).
    Dalam ilmu datamining, clustering menjadi salah satu teknik untuk menganalisis data dengan cara mempartisi data dan 
    membentuknya menjadi suatu kelompok. Konsep dari teknik clustering adalah mengelompokkan sekumpulan data yang memiliki 
    kesamaan menjadi satu kelompok (cluster) dan yang lainnya kedalam kelompok (cluster) lainnya. Clustering telah banyak 
    dimanfaatkan untuk penelitian di berbagai bidang, salah satunya pada pemetaan (pengelompokan) wilayah. 
    """)
    
    st.write("""
    Algoritma yang umum digunakan dalam melakukan clustering ialah algoritma K-Means. 
    Algoritma K-Means merupakan model pembelajaran unsupervised learning (pembelajaran tanpa pengawasan). 
    Model pembelajaran seperti ini digunakan untuk kumpulan data yang belum pernah diberi label atau diklasifikasikan. 
    Algoritma K-Means merupakan algoritma clustering berbasis pembagian jarak. Algoritma ini mudah dimengerti dan 
    diterapkan sehingga banyak digunakan dalam menyelesaikan kasus tentang clustering.
    """)

    st.write("""Kelebihan dari algoritma K-Means ialah dapat diterapkan di banyak jenis pengelompokan dengan kesederhanaan 
    penerapannya dan kompleksitas komputasi yang rendah. Namun, algoritma K-Means juga memiliki beberapa tantangan yang 
    dapat berdampak negatif terhadap kinerja pengelompokannya. Salah satunya adalah dalam proses inisialisasi algoritma. 
    Jumlah cluster dalam suatu kumpulan data harus ditentukan terlebih dahulu secara apriori dan pusat cluster (centroid) 
    awal dipilih secara acak. Hal tersebut akan mempengaruhi performa algoritma terutama jika diterapkan pada kumpulan data 
    yang besar. Menentukan jumlah cluster dan titik centroid yang optimal untuk memulai pengelompokan menjadi hal yang 
    penting karena pemilihan pusat cluster (centroid) awal secara acak terkadang menghasilkan konvergensi lokal yang minimal.
    Hal inilah yang menjadi dasar diperlukan pengulangan proses pemilihan pusat cluster (centroid) awal yang berbeda untuk 
    mendapatkan hasil cluster yang optimal.
    """)

    st.write("""Berdasarkan penjelasan pada paragraf-paragraf sebelumnya, solusi yang dapat ditawarkan untuk mengoptimalkan 
    dan menangani permasalahan penentuan centroid dan jumlah cluster pada algoritma K-Means adalah dengan memanfaatkan 
    kombinasi algoritma Binary Search dan Dynamic K-Means. Binary Search merupakan algoritma pencarian yang efisien dan 
    telah digunakan dalam berbagai konteks, namun belum banyak diaplikasikan dalam konteks optimasi K-Means. 
    Pada penelitian sebelumnya, nilai Davies-Bouldin Index (DBI) dan Silhouette Coefficent juga diperoleh sebagai evaluasi 
    hasil clustering.
    """)

if selected == "Deskripsi":
    st.subheader("Pengertian")
    st.write(""" Di seluruh dunia, breast cancer (kanker payudara) adalah jenis kanker yang paling umum pada wanita dan tertinggi kedua dalam hal angka kematian. Diagnosis kanker payudara dilakukan ketika ditemukan benjolan abnormal (dari pemeriksaan sendiri atau rontgen) atau setitik kecil dari kalsium terlihat (pada x-ray). Setelah benjolan yang mencurigakan ditemukan, dokter akan melakukan diagnosa untuk menentukan apakah itu kanker dan, jika demikian, apakah sudah menyebar ke bagian tubuh yang lain.""")
    st.subheader("Kegunaan Dataset")
    st.write(""" 
    - Data yang digunakan dalam penelitian ini adalah data breast cancer. Dataset ini digunakan untuk mengidentifikasi breast cancer (kanker payudara) seorang pasien termasuk kelas jinak yang memiliki harapan kecil untuk terkena breast cancer (kanker payudara) atau ganas yang dikatakan breast cancer (kanker payudara) yang parah.Oleh karena itu, tujuan dari penelitian ini untuk adalah mengidentifikasi breast cancer secara dini, sehingga data yang digunakan adalah data diagnosis.
    - Dalam data ini terdiri dari 5 atribut penentu apakah kanker tersebut jinak atau ganas, yaitu 
        1. mean_radius
        2. mean_texture
        3. mean_area
        3. mean_perimeter
        4. mean_smoothness
        5. diagnosis  """)
    st.subheader(""" Penyebab dari Breast Cancer""")
    st.write("""Beberapa faktor yang diketahui bisa meningkatkan risiko seseorang terkena kanker payudara adalah:
1.  Usia. Peluang terkena kanker payudara meningkat seiring bertambahnya usia wanita. Hampir 80 persen kanker payudara ditemukan pada wanita di atas usia 50 tahun.

2. Riwayat pribadi kanker payudara. Seorang wanita yang menderita kanker payudara di satu payudara berisiko lebih tinggi terkena kanker di payudara lainnya.

3. Riwayat keluarga kanker payudara. Seorang wanita memiliki risiko lebih tinggi terkena kanker payudara jika ibu, saudara perempuan atau anak perempuannya menderita kanker payudara, terutama pada usia muda (sebelum 40 tahun). Memiliki kerabat lain dengan kanker payudara juga dapat meningkatkan risiko.

4. Faktor genetik. Wanita dengan mutasi genetik tertentu, termasuk perubahan gen BRCA1 dan BRCA2, berisiko lebih tinggi terkena kanker payudara selama hidup mereka. Perubahan gen lainnya juga dapat meningkatkan risiko kanker payudara.

5. Riwayat persalinan dan menstruasi. Semakin tua seorang wanita saat melahirkan anak pertamanya, semakin besar risikonya terkena kanker payudara. Juga berisiko lebih tinggi adalah:
    - Wanita yang menstruasi pertama kali pada usia dini (sebelum 12 tahun)
    - Wanita yang mengalami menopause terlambat (setelah usia 55 tahun)
    - Wanita yang belum pernah memiliki anak

 """)
    st.subheader(""" Tujuan""")
    st.write(""" Analisis ini bertujuan untuk mengamati fitur mana yang paling membantu dalam memprediksi 
    kanker ganas atau jinak dan untuk melihat tren umum yang dapat membantu kita dalam pemilihan model dan pemilihan parameter hiper. Tujuannya adalah untuk mengklasifikasikan apakah kanker payudara tersebut jinak atau ganas. Untuk mencapai ini saya telah menggunakan 
    metode klasifikasi pembelajaran mesin agar sesuai dengan fungsi yang dapat memprediksi kelas diskrit input baru.""")
    st.subheader("Fitur")
    st.markdown(
        """
        Dalam 5 atribut ini, terdapat pengukuran yaitu rata-rata(mean), jarak rata- rata dari titik pusat ke tepi (radius),
        nilai simpangan baku dari tingkat ke abu-abuan (texture), keliling (perimeter), luas area (area), variasi lokasi (smoothness). Berikut ini penjelasan secara rinci setiap fitur yang ada yaitu sebagai berikut :
    
        - mean_radius : rata-rata jarak dari tepi sel breast cancer ke centroid sel tumor tersebut, 

        - mean_texture :  deviasi standar nilai skala abu-abu

        - mean_perimeter :panjang dari keliling sel breast cancer

        - mean_area : luas (jumlah piksel) dari sel breast cancer 

        - mean_smoothness: perbedaan antara panjang garis radial dengan rerata panjang garis radialyang mengelilingi garis radial tersebut.

        - Diagnosis: Diagnosis jaringan breast cancer 
                       0- jinak, 1- ganas
        """
    )

    st.subheader("""Sumber Dataset""")
    st.write("""
    Sumber data di dapatkan melalui website kaggle.com, Berikut merupakan link untuk mengakses sumber dataset.
    <a href="https://www.kaggle.com/datasets/merishnasuwal/breast-cancer-prediction-dataset"> kaggle dataset</a>""", unsafe_allow_html=True)
    
    st.subheader("""Tipe Data""")
    st.write("""
    Tipe data yang di gunakan pada dataset breast cancer ini adalah NUMERICAL.
    """)
    

if selected == "Dataset":
    st.markdown("""<h2 style='text-align: center; color:grey;'> Dataset Breast Cancer Prediction </h1> """, unsafe_allow_html=True)
    df = pd.read_csv('https://raw.githubusercontent.com/Aisyahmsp/clustering_bsdk/main/dataset_produktivitas.csv')
    c1, c2, c3 = st.columns([1,5,1])

    with c1:
        st.write("")

    with c2:
        df

    with c3:
        st.write("")

if selected == "Preprocessing":
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')

    st.write(labels)


if selected == "Clustering":
    df = pd.read_csv('https://raw.githubusercontent.com/Aisyahmsp/clustering_bsdk/main/dataset_produktivitas.csv')


if selected == "Evaluasi":
    df = pd.read_csv('https://raw.githubusercontent.com/Aisyahmsp/clustering_bsdk/main/dataset_produktivitas.csv')
   
if selected == "Referensi":
    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        st.write("") 

    with col3:
        st.write("")

    st.write(""" """)
    
    
    st.write(""" """)

    st.title (f"BIODATA PRIBADI")
    st.markdown(" Aisyah Meta Sari Putri (200411100031)", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size:15px;text-align: left; color: black;'>Dosen Pembimbing 1 : Dr. Bain Khusnul Khotimah, S.T., M.Kom.</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size:15px;text-align: left; color: black;'>Dosen Pembimbing 2 : Dr. Rika Yunitarini, S.T., M.T.</h1>", unsafe_allow_html=True)

    st.title (f"REFERENSI")
    st.subheader("""Sumber Dataset""")
    st.write("""
    Sumber data di dapatkan dari Dinas Ketahanan Pangan dan Pertanian Kabupaten Sumenep Tahun 2020.
   """)
