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

st.markdown("""<h1 style='text-align: center;'> Breast Cancer Prediction Dataset </h1> """, unsafe_allow_html=True)
# 1. as sidevar menu
with st.sidebar:
    st.markdown("<h1 style='font-size:15px;text-align: center; color: black;'>Normalita Eka Ariyanti_200411100084</h1><h1 style='font-size:15px;text-align: center; color: black;'>Aisyatur Radiah_200411100116</h1>", unsafe_allow_html=True)
    selected = option_menu(
        menu_title="Clustering", #required
        options=["Beranda", "Description", "Dataset", "Preprocessing", "Clustering", "Evaluation", "Referensi"], #required
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
        img = Image.open('breast cancer.jpg')
        st.image(img, use_column_width=False, width=300)

    with col3:
        st.write("")

    st.write(""" """)
    
    
    st.write(""" """)

    st.write("""
    Di seluruh dunia, breast cancer adalah jenis kanker yang paling umum pada wanita dan tertinggi kedua
    dalam hal angka kematian. Diagnosis kanker payudara dilakukan ketika ditemukan benjolan abnormal
    (dari pemeriksaan sendiri atau rontgen) atau setitik kecil dari kalsium terlihat (pada x-ray). 
    Setelah benjolan yang mencurigakan ditemukan, dokter akan melakukan diagnosa untuk menentukan
    apakah itu kanker dan, jika demikian, apakah sudah menyebar ke bagian tubuh yang lain.

    Breast Cancer/ Kanker Payudara adalah kondisi ketika sel kanker terbentuk di jaringan payudara.
    Breast Cancer bisa terbentuk di kelenjar yang menghasilkan susu (lobulus), 
    atau di saluran (duktus) yang membawa air susu dari kelenjar ke puting payudara. 
    Kanker juga bisa terbentuk di jaringan lemak atau jaringan ikat di dalam payudara. 
    """)
    
    st.write("""
    Kanker payudara terbentuk saat sel-sel di dalam payudara tumbuh tidak normal
    dan tidak terkendali. Sel tersebut umumnya membentuk tumor yang terasa seperti benjolan. 
    Meski biasanya terjadi pada wanita, kanker payudara juga bisa menyerang pria.
    """)

    st.write("""breast cancer adalah suatu kanker dimana bertumbuhnya serta berkembangnya sebuah sel-sel jaringan yang mengerikan 
    yang tumbuh di area payudara.Kanker payudara merupakan kanker yang sering terjadi kedua di dunia dan kanker yang paling 
    sering dirasakan oleh diantara wanita dengan perkiraan 1,67 juta kasus kanker baru yang didiagnosis pada tahun 2012
     (25% dari semua kanker).
    """)

    st.write("""Secara umum, semakin tinggi tingkat keparahan kanker (semakin berkembang dan menyebar), 
    semakin kecil kemungkinan pengobatan yang dilakukan dapat menyembuhkan kanker payudara. 
    Namun, pengobatan yang dilakukan dapat memperlambat perkembangan kanker.
    """)

if selected == "Description":
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


if selected == "Evaluation":
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
   
    st.subheader("""""")
    st.write("""Sumber data di dapatkan melalui github, Berikut merupakan link untuk mengakses sumber dataset dari github.
