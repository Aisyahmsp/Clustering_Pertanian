import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import altair as alt
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import mode
from sklearn.metrics.pairwise import pairwise_distances
from itertools import combinations

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
    Dinas Ketahanan Pangan dan Pertanian Kabupaten Sumenep Tahun 2022. Data dapat diakses melalui laman""")
    st.write(
    """<a href="https://raw.githubusercontent.com/Aisyahmsp/Clustering_Pertanian/main/Informasi%20Fitur.csv">dataset produktivitas pertanian</a>""",
    unsafe_allow_html=True)
    
    st.subheader(""" Informasi Detail Fitur""")
    fitur = pd.read_csv('https://raw.githubusercontent.com/Aisyahmsp/Clustering_Pertanian/main/Informasi%20Fitur.csv')
    fitur

if selected == "Dataset":
    st.markdown("""<h2 style='text-align: center; color:grey;'> Dataset Produktivitas Pertanian Kabupaten Sumenep Tahun 2022 </h1> """, unsafe_allow_html=True)
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
                st.write("Maksimum iterasi telah dicapai.")
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
        # Mengelompokkan data berdasarkan cluster dan menampilkan kelompok tani di setiap clusternya
        for cluster in sorted(data_with_group['Cluster'].unique()):
            kelompok_tani = data_with_group[data_with_group['Cluster'] == cluster]['Kelompok Tani'].tolist()
            jumlah_kelompok = len(kelompok_tani)
            st.subheader(f"\nCluster {cluster} terdiri dari {jumlah_kelompok} kelompok tani berikut:")
            for kelompok in kelompok_tani:
                st.write(f"- {kelompok}")
        

    with Nilai_DBI:
        st.subheader("Nilai DBI")
        # Hitung SSW untuk iterasi terakhir
        SSW = np.zeros(num_clusters)
        for i in range(num_clusters):
            cluster_data = jarak_data_centroid_baru[label == i+1,0]
            m_i = len(cluster_data)
            if m_i==0:
              SSW[i]=0
            else:
              SSW[i] = np.sum(cluster_data) / m_i
        
        # Menghitung SSB dengan jarak Hamming untuk fitur pertama hingga ketiga dan jarak Euclidean untuk fitur lainnya
        SSB = np.zeros((num_clusters, num_clusters))
        
        for i, j in combinations(range(num_clusters), 2):
            hamming_distance = np.sum(new_centroids[i, 7:] != new_centroids[j, 7:])
            euclidean_distance = np.sqrt(np.sum((new_centroids[i, :7] - new_centroids[j, :7]) ** 2))
            SSB[i, j] = hamming_distance + euclidean_distance
            SSB[j, i] = SSB[i, j]  # Karena matriks simetris
        
        # Menghitung Rasio R_ij
        R_ij = np.zeros((num_clusters, num_clusters))
        for i, j in combinations(range(num_clusters), 2):
            SSB_ij = SSB[i, j]
            R_ij[i, j] = (SSW[i] + SSW[j]) / SSB_ij
            R_ij[j, i] = R_ij[i, j]  # Karena matriks simetris
        
        # Hitung DBI
        DBI = (np.sum(np.max(R_ij, axis=1)))/num_clusters
        st.write(f"Nilai DBI K-Means dengan nilai k = {num_clusters}: ", DBI)

    with Nilai_Silhouette:
        st.subheader("Nilai Silhouette Coefficent")
        def hitung_intra_cluster(data_asli, label, centroids):
            num_clusters = len(centroids)
            intra_cluster = []
            for i in range(num_clusters):
                cluster_data = data_asli[label == (i + 1)]
                if len(cluster_data) > 0:
                    distances = (hitung_jarak(cluster_data, centroids))
                    intra_cluster.append(distances.mean())
                else:
                    intra_cluster.append(0)
            return np.sum(intra_cluster)
        
        def hitung_inter_cluster(centroids):
            inter_cluster = 0
            for (i, j) in combinations(range(len(centroids)), 2):
                hamming_distance = np.sum(centroids[i, 7:] != centroids[j, 7:])
                euclidean_distance = np.sqrt(np.sum((centroids[i, :7] - centroids[j, :7]) ** 2))
                inter_cluster += hamming_distance + euclidean_distance
            return inter_cluster
        if len(hasil_iterasi) >= 2:
            _, label_intra_baru, centroids_intra_baru = hasil_iterasi[-1]
            intra_baru = hitung_intra_cluster(data_asli, label_intra_baru, centroids_intra_baru)
            inter_baru = hitung_inter_cluster(centroids_intra_baru)
        # Fungsi untuk menghitung silhouette coefficient
        def compute_silhouette_coefficient(intra, inter):
            if intra == 0 and inter == 0:
                return 0
            else:
                return (inter - intra) / max(intra, inter)
        
        # Menghitung silhouette coefficient
        silhouette_coefficient = compute_silhouette_coefficient(intra_baru, inter_baru)
        
        # Menampilkan hasil
        st.write(f"Nilai Silhouette Coefficient K-Means dengan nilai k = {num_clusters}:", silhouette_coefficient)


if selected == "Clustering BSDK":
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
        # Memasukkan nilai jumlah cluster
        num_clusters = st.number_input("Masukkan jumlah cluster:", min_value=1, max_value=len(data), step=1, value=3)
        # Mencari nilai min dan max untuk setiap fitur
        min_values = data.min()
        max_values = data.max()
        # Mendapatkan nilai range untuk setiap fitur
        ranges = (max_values - min_values) / num_clusters
        # Menentukan nilai centroid awal untuk setiap fitur
        centroids_binary_search = []
        for k in range(1, num_clusters + 1):
            centroid_row = []
            for i in range(len(min_values)):
                if i in [7, 8, 9]:  # Fitur ke-8, 9, dan 10
                    centroid_value = round(min_values[i] + (k - 1) * ranges[i])
                else:
                    centroid_value = min_values[i] + (k - 1) * ranges[i]
                centroid_row.append(centroid_value)
            centroids_binary_search.append(centroid_row)
        
        # Konversi ke DataFrame untuk tampilan yang lebih baik
        centroids = pd.DataFrame(centroids_binary_search, columns=data.columns)
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
                        if k < 7:  # Kolom 1-7 menggunakan Euclidean Distance
                            total_euclidean += (data[i][k] - centroid[j][k])**2
                        else:  # Kolom 8, 9, 10 menggunakan Hamming Distance
                            if data[i][k] != centroid[j][k]:
                                total_hamming += 1
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
        def hitung_intra_cluster(data_asli, label, centroids):
            num_clusters = len(centroids)
            intra_cluster = []
            for i in range(num_clusters):
                cluster_data = data_asli[label == (i + 1)]
                if len(cluster_data) > 0:
                    distances = (hitung_jarak(cluster_data, centroids))
                    intra_cluster.append(distances.mean())
                else:
                    intra_cluster.append(0)
            return np.sum(intra_cluster)
        
        def hitung_inter_cluster(centroids):
            inter_cluster = 0
            for (i, j) in combinations(range(len(centroids)), 2):
                hamming_distance = np.sum(centroids[i, 7:] != centroids[j, 7:])
                euclidean_distance = np.sqrt(np.sum((centroids[i, :7] - centroids[j, :7]) ** 2))
                inter_cluster += hamming_distance + euclidean_distance
            return inter_cluster
        while True:
          if len(hasil_iterasi) >= 2:
            _, label_intra_lama, centroids_intra_lama = hasil_iterasi[-2]
            _, label_intra_baru, centroids_intra_baru = hasil_iterasi[-1]
            intra_lama = hitung_intra_cluster(data_asli, label_intra_lama, centroids_intra_lama)
            intra_baru = hitung_intra_cluster(data_asli, label_intra_baru, centroids_intra_baru)
            inter_lama = hitung_inter_cluster(centroids_intra_lama)
            inter_baru = hitung_inter_cluster(centroids_intra_baru)
        
            if intra_baru < intra_lama and inter_baru > inter_lama:
                st.write("intra_baru < intra_lama dan inter_baru > inter_lama")
                st.write(f"Menambah jumlah cluster menjadi {num_clusters+1}. Mulai iterasi ulang.")
            else:
                st.write(f"Jumlah cluster optimal: {num_clusters}")
                break
          else:
            st.write("Tidak cukup iterasi untuk menghitung intra-cluster dan inter-cluster.")
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
        # Mengelompokkan data berdasarkan cluster dan menampilkan kelompok tani di setiap clusternya
        for cluster in sorted(data_with_group['Cluster'].unique()):
            kelompok_tani = data_with_group[data_with_group['Cluster'] == cluster]['Kelompok Tani'].tolist()
            jumlah_kelompok = len(kelompok_tani)
            st.subheader(f"\nCluster {cluster} terdiri dari {jumlah_kelompok} kelompok tani berikut:")
            for kelompok in kelompok_tani:
                st.write(f"- {kelompok}")
                
    with Nilai_DBI:
        st.subheader("Nilai DBI")
        # Hitung SSW untuk iterasi terakhir
        SSW = np.zeros(num_clusters)
        for i in range(num_clusters):
            cluster_data = jarak_data_centroid_baru[label == i+1,0]
            m_i = len(cluster_data)
            if m_i==0:
              SSW[i]=0
            else:
              SSW[i] = np.sum(cluster_data) / m_i
        # Menghitung SSB dengan jarak Hamming untuk fitur pertama hingga ketiga dan jarak Euclidean untuk fitur lainnya
        SSB = np.zeros((num_clusters, num_clusters))
        for i, j in combinations(range(num_clusters), 2):
            hamming_distance = np.sum(new_centroids[i, 7:] != new_centroids[j, 7:])
            euclidean_distance = np.sqrt(np.sum((new_centroids[i, :7] - new_centroids[j, :7]) ** 2))
            SSB[i, j] = hamming_distance + euclidean_distance
            SSB[j, i] = SSB[i, j]  # Karena matriks simetris
        # Menghitung Rasio R_ij
        R_ij = np.zeros((num_clusters, num_clusters))
        for i, j in combinations(range(num_clusters), 2):
            SSB_ij = SSB[i, j]
            R_ij[i, j] = (SSW[i] + SSW[j]) / SSB_ij
            R_ij[j, i] = R_ij[i, j]  # Karena matriks simetris
        
        # Hitung DBI
        DBI = (np.sum(np.max(R_ij, axis=1)))/num_clusters
        st.write(f"Nilai DBI BSDK dengan nilai k = {num_clusters}: ", DBI)
        
    with Nilai_Silhouette:
        st.subheader("Nilai Silhouette Coefficent")
        # Fungsi untuk menghitung silhouette coefficient
        def compute_silhouette_coefficient(intra, inter):
            if intra == 0 and inter == 0:
                return 0
            else:
                return (inter - intra) / max(intra, inter)
        # Menghitung silhouette coefficient
        silhouette_coefficient = compute_silhouette_coefficient(intra_baru, inter_baru)
        # Menampilkan hasil
        st.write(f"Nilai Silhouette Coefficient BSDK dengan k = {num_clusters}:", silhouette_coefficient)
   
if selected == "Referensi":
    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        st.write("") 

    with col3:
        st.write("")

    st.write(""" """)
    
    
    st.write(""" """)

    st.title (f"BIODATA")
    st.markdown("<h1 style='font-size:15px;text-align: left; color: black;'>Aisyah Meta Sari Putri (200411100031)</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size:15px;text-align: left; color: black;'>Dosen Pembimbing 1 : Dr. Bain Khusnul Khotimah, S.T., M.Kom.</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size:15px;text-align: left; color: black;'>Dosen Pembimbing 2 : Dr. Rika Yunitarini, S.T., M.T.</h1>", unsafe_allow_html=True)

    st.title (f"REFERENSI")
    st.subheader("""Sumber Dataset""")
    st.write("""
    Sumber data di dapatkan dari Dinas Ketahanan Pangan dan Pertanian Kabupaten Sumenep Tahun 2022.
   """)
    st.subheader("""Sumber Jurnal""")
    st.write("""
   -  M. Y. Nurzaman and B. Nurina Sari, “Implementasi K-Means Clustering Dalam Pengelompokan Banyaknya Jumlah Petani Berdasarkan Kecamatan Di Provinsi Jawa Barat,” Jurnal Teknik Informatika dan Sistem Informasi, vol. 10, no. 3, 2023, [Online]. Available: http://jurnal.mdp.ac.id
   - Sekar Setyaningtyas, B. Indarmawan Nugroho, and Z. Arif, “Tinjauan Pustaka Sistematis Pada Data Mining : Studi Kasus Algoritma K-Means Clustering,” Jurnal Teknoif Teknik Informatika Institut Teknologi Padang, vol. 10, no. 2, pp. 52–61, Oct. 2022, doi: 10.21063/jtif.2022.v10.2.52-61.
   - N. A. Khairani and E. Sutoyo, “Application of K-Means Clustering Algorithm for Determination of Fire-Prone Areas Utilizing Hotspots in West Kalimantan Province,” International Journal of Advances in Data and Information Systems, vol. 1, no. 1, pp. 9–16, Apr. 2020, doi: 10.25008/ijadis.v1i1.13.
   - P. Trisnawati and A. I. Purnamasari, “Penerapan Pengelompokkan Produktivitas Hasil Pertanian Menggunakan Algoritma K-Means,” Infotek: Jurnal Informatika dan Teknologi, vol. 6, no. 2, pp. 249–257, 2023.
   - H. Santoso, H. Magdalena, and H. Wardhana, “Aplikasi Dynamic Cluster pada K-Means BerbasisWeb untuk Klasifikasi Data Industri Rumahan,” MATRIK: Jurnal Manajemen, Teknik Informatika dan Rekayasa Komputer, vol. 21, no. 3, pp. 541–554, 2022.
   - I. Reisandi, Daryana, F. Sri Mulyati, and M. Fauzi, “Implementasi Clustering K-Means Terhadap Penilaian Kinerja Karyawan PT.XYZ,” Jurnal Sosial dan Teknologi, vol. 1, no. 8, Aug. 2021, [Online]. Available: http://sostech.greenvest.co.id
   - J. Hutagalung, N. L. W. S. R. Ginantra, G. W. Bhawika, W. G. S. Parwita, A. Wanto, and P. D. Panjaitan, “Covid-19 Cases and Deaths in Southeast Asia Clustering using K-Means Algorithm,” in Journal of Physics: Conference Series, IOP Publishing Ltd, Feb. 2021. doi: 10.1088/1742-6596/1783/1/012027.
   - K. Ariasa, I. Gede, A. Gunadi, and I. Made Candiasa, “Optimasi Algoritma Klaster Dinamis Pada K-Means Dalam Pengelompokan Kinerja Akademik Mahasiswa (Studi Kasus: Universitas Pendidikan Ganesha),” JANAPATI, vol. 9, no. 2, 2020.
   - G. Akbari and Y. Kerlooza, “Peningkatan Hasil Cluster Menggunakan Algoritma Dynamic K-means dan K-means Binary Search Centroid,” Jurnal Tata Kelola dan Kerangka Kerja Teknologi Informasi, vol. 4, no. 2, pp. 25–33, 2018.
   - A. Mahmudan, “Clustering of District or City in Central Java Based Covid-19 Case Using K-Means Clustering (Pengelompokan Kabupaten/Kota di Jawa Tengah Berdasarkan Kasus Covid-19 Menggunakan K-Means Clustering),” Jurnal Matematika, Statistika, dan Komputasi, vol. 17, no. 1, pp. 1–13, 2020, doi: 10.20956/jmsk.v%vi%i.10727.
   - M. Andryan, M. Faisal, and R. Kusumawati, “K-Means Binary Search Centroid With Dynamic Cluster for Java Island Health Clustering,” Jurnal Riset Informatika, 2023, [Online]. Available: https://api.semanticscholar.org/CorpusID:259591541
   """)
