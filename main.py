import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import altair as alt
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import mode

# display
st.set_page_config(page_title="Clustering", page_icon='logo.jpeg')

st.markdown("""<h1 style='text-align: center;'> Pengelompokan Produktivitas Pertanian </h1> """, unsafe_allow_html=True)
# 1. as sidevar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Pilihan Menu", #required
        options=["Beranda", "Deskripsi", "Dataset", "Preprocessing", "Clustering K-Means", "Clustering BSDK", "Referensi"], #required
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
            "--hover-color": "#F4A261",
        },
        "nav-link-selected": {"background-color": "#E2725B"}
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
    Dinas Ketahanan Pangan dan Pertanian Kabupaten Sumenep Tahun 2020. Data dapat diakses melalui laman""")
    st.write(
    """<a href="https://raw.githubusercontent.com/Aisyahmsp/Clustering_Pertanian/main/Informasi%20Fitur.csv">dataset produktivitas pertanian</a>""",
    unsafe_allow_html=True)
    
    st.subheader(""" Informasi Detail Fitur""")
    fitur = pd.read_csv('https://raw.githubusercontent.com/Aisyahmsp/Clustering_Pertanian/main/Informasi%20Fitur.csv')
    fitur

if selected == "Dataset":
    st.markdown("""<h2 style='text-align: center; color:grey;'> Dataset Produktivitas Pertanian Kabupaten Sumenep Tahun 2020 </h1> """, unsafe_allow_html=True)
    data = pd.read_csv('https://raw.githubusercontent.com/Aisyahmsp/clustering_bsdk/main/dataset_produktivitas.csv')
    c1, c2, c3 = st.columns([1,5,1])

    with c1:
        st.write("")

    with c2:
        data

    with c3:
        st.write("")

if selected == "Preprocessing":
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    rumus = Image.open('rumus_norm.jpg')
    st.image(rumus, use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    
    st.subheader('Hasil Normalisasi Data')
    data = pd.read_csv('https://raw.githubusercontent.com/Aisyahmsp/clustering_bsdk/main/dataset_produktivitas.csv')
    # MENGHAPUS FITUR YANG TIDAK RELEVAN
    # Tentukan daftar fitur yang ingin dihapus
    delete_fitur = ['No', 'Tanggal', 'Kecamatan', 'Desa']
    # Gunakan metode drop untuk menghapus fitur dari DataFrame
    data_clean = data.drop(delete_fitur, axis=1)
    # Tampilkan DataFrame setelah fitur dihapus
    data = data_clean
    # Tentukan fitur yang ingin dihapus sementara
    fitur_poktan = data['Kelompok Tani']
    # Hapus fitur desa dari DataFrame
    data.drop('Kelompok Tani', axis=1, inplace=True)

    # TRANSFORMASI DATA
    # Fitur-fitur yang ingin diperiksa
    fitur_list = ['Komoditas', 'Varietas', 'Jenis OPT']
    
    # Mendapatkan daftar nilai unik yang sudah diurutkan dan mengganti nilai dalam kolom dengan angka sesuai urutan nilai unik
    for fitur in fitur_list:
        unique_values = sorted(data[fitur].unique())
        data[f'{fitur}_Kode'] = data[fitur].map({value: i + 1 for i, value in enumerate(unique_values)})
    
    # Menampilkan DataFrame dengan kolom baru untuk setiap fitur
    fitur_kode_columns = [f'{fitur}_Kode' for fitur in fitur_list]
    # Drop fitur 'Komoditas', 'Varietas', dan 'Jenis Opt'
    data.drop(['Komoditas', 'Varietas', 'Jenis OPT'], axis=1, inplace=True)
    # Rename fitur menjadi 'Komoditas_Kode', 'Varietas_Kode', dan 'Jenis_Opt_Kode'
    data.rename(columns={'Komoditas_Kode': 'Komoditas', 'Varietas_Kode': 'Varietas', 'Jenis OPT_Kode': 'Jenis_OPT'}, inplace=True)

    # MISSING VALUE
    data['Luas Terserang (Ha)'] = pd.to_numeric(data['Luas Terserang (Ha)'], errors='coerce')
    data['Intensitas (%)'] = pd.to_numeric(data['Intensitas (%)'], errors='coerce')
    # Menampilkan nilai mean dari masing-masing fitur
    mean_values = data.mean()
    # Mengganti nilai null dengan nilai mean
    data = data.fillna(mean_values)

    # NORMALISASI DATA
    # Fitur-fitur yang ingin dinormalisasi
    fitur_list = ['Luas Tanam (ha)', 'Stadia/Umur Tanaman (hst)', 'Pupuk Bersubsidi Organik dan Anorganik (Ton)',
                  'Luas Terserang (Ha)', 'Intensitas (%)','Luas Waspada (Ha)','Hasil Panen (ton)']
    # Membuat objek MinMaxScaler
    scaler = MinMaxScaler()
    # Melakukan normalisasi Min-Max Scalar hanya pada fitur yang dipilih
    data[fitur_list] = scaler.fit_transform(data[fitur_list])
    # Menampilkan dataframe setelah normalisasi
    data

if selected == "Clustering K-Means":
    data = pd.read_csv('https://raw.githubusercontent.com/Aisyahmsp/clustering_bsdk/main/dataset_produktivitas.csv')
    # MENGHAPUS FITUR YANG TIDAK RELEVAN
    # Tentukan daftar fitur yang ingin dihapus
    delete_fitur = ['No', 'Tanggal', 'Kecamatan', 'Desa']
    # Gunakan metode drop untuk menghapus fitur dari DataFrame
    data_clean = data.drop(delete_fitur, axis=1)
    # Tampilkan DataFrame setelah fitur dihapus
    data = data_clean
    # Tentukan fitur yang ingin dihapus sementara
    fitur_poktan = data['Kelompok Tani']
    # Hapus fitur desa dari DataFrame
    data.drop('Kelompok Tani', axis=1, inplace=True)

    # TRANSFORMASI DATA
    # Fitur-fitur yang ingin diperiksa
    fitur_list = ['Komoditas', 'Varietas', 'Jenis OPT']
    
    # Mendapatkan daftar nilai unik yang sudah diurutkan dan mengganti nilai dalam kolom dengan angka sesuai urutan nilai unik
    for fitur in fitur_list:
        unique_values = sorted(data[fitur].unique())
        data[f'{fitur}_Kode'] = data[fitur].map({value: i + 1 for i, value in enumerate(unique_values)})
    
    # Menampilkan DataFrame dengan kolom baru untuk setiap fitur
    fitur_kode_columns = [f'{fitur}_Kode' for fitur in fitur_list]
    # Drop fitur 'Komoditas', 'Varietas', dan 'Jenis Opt'
    data.drop(['Komoditas', 'Varietas', 'Jenis OPT'], axis=1, inplace=True)
    # Rename fitur menjadi 'Komoditas_Kode', 'Varietas_Kode', dan 'Jenis_Opt_Kode'
    data.rename(columns={'Komoditas_Kode': 'Komoditas', 'Varietas_Kode': 'Varietas', 'Jenis OPT_Kode': 'Jenis_OPT'}, inplace=True)

    # MISSING VALUE
    data['Luas Terserang (Ha)'] = pd.to_numeric(data['Luas Terserang (Ha)'], errors='coerce')
    data['Intensitas (%)'] = pd.to_numeric(data['Intensitas (%)'], errors='coerce')
    # Menampilkan nilai mean dari masing-masing fitur
    mean_values = data.mean()
    # Mengganti nilai null dengan nilai mean
    data = data.fillna(mean_values)

    # NORMALISASI DATA
    # Fitur-fitur yang ingin dinormalisasi
    fitur_list = ['Luas Tanam (ha)', 'Stadia/Umur Tanaman (hst)', 'Pupuk Bersubsidi Organik dan Anorganik (Ton)',
                  'Luas Terserang (Ha)', 'Intensitas (%)','Luas Waspada (Ha)','Hasil Panen (ton)']
    # Membuat objek MinMaxScaler
    scaler = MinMaxScaler()
    # Melakukan normalisasi Min-Max Scalar hanya pada fitur yang dipilih
    data[fitur_list] = scaler.fit_transform(data[fitur_list])

    Hasil_Clustering, Rincian_Cluster, Nilai_DBI, Nilai_Silhouette = st.tabs(["Hasil Clustering", "Rincian Cluster", "Nilai DBI", "Nilai Silhouette"])
    
    with Hasil_Clustering:
        # Memasukkan jumlah cluster menggunakan Streamlit
        num_clusters = st.number_input("Masukkan jumlah cluster:", min_value=1, max_value=len(data), step=1, value=3)
        st.subheader("""Hasil Clustering""")
        # Mendapatkan nilai centroid awal dari n baris pertama data secara acak
        centroids = data.sample(n=num_clusters, random_state=42)
        centroid = centroids.values
        data_asli = data.values
        # Mendefinisikan fungsi untuk menghitung jarak antara data dan centroid
        def hitung_jarak(data, centroid):
            jarak = []
            for i in range(len(data)):
                jarak_centroid = []
                for j in range(len(centroid)):
                    total_euclidean = 0
                    total_hamming = 0
                    for k in range(len(data[i])):
                        if k in range(7,10):
                          # Kolom 8, 9, 10 menggunakan Hamming Distance
                            if data[i][k] != centroid[j][k]:
                                total_hamming += 1
                            else:
                              total_hamming = 0
                        else:  # Kolom 1-7 menggunakan Euclidean Distance
                            total_euclidean += (data[i][k] - centroid[j][k])**2
                    total_euclidean = np.sqrt(total_euclidean)  # Akar dari total Euclidean Distance
                    total_jarak = total_euclidean + total_hamming  # Total jarak = Euclidean + Hamming
                    jarak_centroid.append(total_jarak)
                jarak.append(jarak_centroid)
            return np.array(jarak)
        
        # Menghitung jarak antara data asli dan centroid
        jarak_data_centroid = hitung_jarak(data_asli, centroid)
        # Memilih jarak terkecil untuk setiap baris
        jarak_terkecil = np.argmin(jarak_data_centroid, axis=1)
        # Menampilkan label berdasarkan jarak terkecil
        label = jarak_terkecil + 1
        # Fungsi untuk mengupdate centroid berdasarkan label
        def update_centroid(data_asli, label, num_clusters, centroids):
            # Mengelompokkan data berdasarkan label cluster
            data_cluster = {}
            for i in range(1, num_clusters + 1):
                data_cluster[i] = data_asli[label == i]
        
            # Memperbarui centroid
            new_centroids = []
        
            # Loop untuk setiap cluster
            for i in range(1, num_clusters + 1):
                cluster_data = data_cluster[i]
                centroid_row = []
        
                # Jika klaster kosong, gunakan centroid sebelumnya
                if len(cluster_data) == 0:
                   centroid_row = centroid[i-1]  # Gunakan centroid sebelumnya
                else:
                    # Hitung nilai rata-rata untuk kolom 1 sampai 7
                    for j in range(7):
                        mean_value = np.mean(cluster_data[:, j])
                        centroid_row.append(mean_value)
        
                    # Hitung nilai yang paling sering muncul untuk kolom 8, 9, dan 10
                    for j in range(7, 10):
                        mode_values = mode(cluster_data[:, j])[0]  # Mengambil nilai yang paling sering muncul
                        if isinstance(mode_values, np.ndarray):  # Memeriksa apakah ada lebih dari satu mode
                            mode_value = mode_values[0]  # Mengambil nilai pertama dari hasil mode
                        else:
                            mode_value = mode_values  # Jika hanya ada satu mode, gunakan nilai tersebut
                        centroid_row.append(mode_value)
        
                # Tambahkan baris centroid ke daftar centroid baru
                new_centroids.append(centroid_row)
        
            return np.array(new_centroids)
            # Inisialisasi variabel
            max_iter = 100
            iter_count = 0
            prev_label = None
            
            # List untuk menyimpan hasil iterasi
            hasil_iterasi = []
            
            # Iterasi K-Means
            while True:
                # Increment iterasi
                iter_count += 1
            
                # Update centroid
                new_centroids = update_centroid(data_asli, label, num_clusters,centroids)
            
                # Hitung jarak antara data dan centroid baru
                jarak_data_centroid_baru = hitung_jarak(data_asli, new_centroids)
            
                # Memilih jarak terkecil untuk setiap baris
                jarak_terkecil_baru = np.argmin(jarak_data_centroid_baru, axis=1)
            
                # Mengecek konvergensi dengan membandingkan hasil label dengan iterasi sebelumnya
                if prev_label is not None and np.array_equal(jarak_terkecil_baru +1, prev_label):
                    st.write(f"Konvergensi dicapai setelah iterasi ke-{iter_count}.")
                    hasil_iterasi.append((iter_count, label, new_centroids))
                    break
            
                # Update label
                label = jarak_terkecil_baru +1
            
                # Simpan label untuk iterasi berikutnya
                prev_label = np.copy(label)
            
                # Menyimpan hasil iterasi
                hasil_iterasi.append((iter_count, label, new_centroids))
            
                # Memeriksa apakah sudah mencapai maksimum iterasi
                if iter_count >= max_iter:
                    break
                # Menggabungkan data asli dengan label clustering terakhir
                data_with_group = data.copy()
                data_with_group['Cluster'] = label
                data_with_group.insert(0, 'Kelompok Tani', fitur_poktan)
                
                # Menampilkan hasil clustering terakhir dengan nama kelompok tani
                st.write("Hasil Clustering Terakhir dengan Nama Kelompok Tani dan labelnya:")
                data_with_group

    with Rincian_Cluster:
        st.subheader("Rincian Hasil Cluster")
        # Menggabungkan data asli dengan label clustering terakhir
        data_with_group = data.copy()
        data_with_group['Cluster'] = label
        data_with_group.insert(0, 'Kelompok Tani', fitur_poktan)
        # Mengelompokkan data berdasarkan cluster dan menampilkan kelompok tani di setiap clusternya
        for cluster in sorted(data_with_group['Cluster'].unique()):
            kelompok_tani = data_with_group[data_with_group['Cluster'] == cluster]['Kelompok Tani'].tolist()
            jumlah_kelompok = len(kelompok_tani)
            st.subheader(f"\nCluster {cluster} terdiri dari {jumlah_kelompok} kelompok tani berikut:")
            for kelompok in kelompok_tani:
                st.write(f"- {kelompok}")
        

    with Nilai_DBI:
        st.write("")

    with Nilai_Silhouette:
        st.write("")
    


if selected == "Clustering BSDK":
    df = pd.read_csv('https://raw.githubusercontent.com/Aisyahmsp/clustering_bsdk/main/dataset_produktivitas.csv')
    st.subheader('Nilai DBI')
    st.subheader('Nilai Sillhouette')
    st.subheader('Hasil Clustering')
   
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
