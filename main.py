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
        st.image(img, use_column_width=False, width=250)

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
    st.subheader("Permasalahan")
    st.write("""Ada beberapa kelemahan dalam penggunaan algoritma K-Means pada pengelompokan produktivitas pertanian 
    di Kabupaten Sumenep. Salah satu kelemahan dari algoritma ini ialah jumlah cluster dalam suatu kumpulan data harus 
    ditentukan terlebih dahulu secara apriori. Tidak ada ketentuan khusus untuk menentukan jumlah cluster yang ingin 
    dibentuk. Selain itu, pemilihan pusat cluster (centroid) awal dilakukan secara acak. Hal tersebut akan mempengaruhi 
    performa algoritma terutama jika diterapkan pada kumpulan data yang besar. Oleh karena itu, dibutuhkan algoritma 
    tambahan yang dapat melengkapi kelemahan dari algoritma K-Means untuk melakukan pengelompokan pada produktivitas 
    pertanian di Kabupaten Sumenep.""")
    
    st.subheader("Solusi")
    st.write(""" 
    Solusi permasalahan ini melibatkan pengembangan dan implementasi algoritma Binary Search Dynamic K-Means Clustering 
    untuk mengelompokkan produktivitas pertanian di Kabupaten Sumenep. Binary Search digunakan untuk mengurutkan centroid 
    dalam penentuan centroid awal, sehingga centroid yang dihasilkan bisa optimal. Sementara Dynamic K-Means digunakan 
    untuk mengatasi inisialisasi jumlah cluster secara dinamis. Sehingga nantinya didapatkan jumlah cluster paling optimal 
    pada data tersebut. Penggabungan kedua algoritma ini memungkinkan identifikasi kelompok pertanian yang serupa secara 
    otomatis, yang dapat membantu pemerintah dan pemangku kepentingan dalam mengambil keputusan yang lebih baik terkait 
    pengembangan sektor pertanian di daerah ini.  """)
    
    st.subheader("""Sumber Data""")
    st.write(""" Data yang digunakan merupakan data produktivitas pertanian yang diperoleh dari 
    Dinas Ketahanan Pangan dan Pertanian Kabupaten Sumenep Tahun 2020. Data dapat diakses melalui laman <a href="https://www.kaggle.com/datasets/merishnasuwal/breast-cancer-prediction-dataset"> dataset produktivitas pertanian</a>""")
    
    st.subheader(""" Informasi Detail Fitur""")
    fitur = pd.read_csv('https://raw.githubusercontent.com/Aisyahmsp/clustering_bsdk/main/dataset_produktivitas.csv')
    

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
