# Prediksi Kehadiran Anggota Pelatihan
## PT Cahaya Ladara Nusantara

Ini adalah aplikasi prediksi kehadiran anggota pada kegiatan pelatihan makanan viral Sushi dan Onigiri yang diselenggarakan oleh PT Cahaya Ladara Nusantara. Aplikasi ini dibangun sebagai bagian dari skripsi dengan menggunakan algoritma Random Forest dan di-deploy sebagai web app interaktif berbasis Streamlit.

---

## Tentang Proyek

Judul skripsi: Prediksi Kehadiran Anggota Menggunakan Algoritma Random Forest Berdasarkan Data Riwayat Pelatihan Makanan Viral Sushi dan Onigiri di PT Cahaya Ladara Nusantara

Program Studi: Sistem Informasi, Universitas Dirgantara Marsekal Suryadarma

Metode yang digunakan adalah Knowledge Discovery in Database (KDD) dengan lima tahapan: seleksi data, praproses, transformasi, data mining, dan evaluasi. Model yang dibangun adalah Random Forest Classifier dari library scikit-learn.

Data yang digunakan berasal dari dua event pelatihan yang diselenggarakan pada Minggu, 09 November 2025 (Event 1) dan Minggu, 16 November 2025 (Event 2), dengan total 300 data peserta.

---

## Fitur Aplikasi

Aplikasi terdiri dari empat halaman utama:

Prediksi Kehadiran: Masukkan data peserta secara manual, klik tombol prediksi, dan dapatkan hasil prediksi Hadir atau Tidak Hadir beserta probabilitasnya. Hasil bisa diunduh dalam format CSV.

Dashboard: Menampilkan ringkasan statistik dataset dan lima grafik interaktif, mulai dari distribusi kehadiran per event, distribusi usia, distribusi jarak tempat tinggal, hingga perbandingan gender.

Evaluasi Model: Menampilkan metrik evaluasi lengkap (akurasi, presisi, recall, F1-score), visualisasi confusion matrix, grafik feature importance, dan laporan PDF yang bisa diunduh.

Log Prediksi: Menyimpan riwayat prediksi selama sesi berlangsung dan memungkinkan ekspor ke CSV.

---

## Hasil Model

Model Random Forest yang dilatih menghasilkan akurasi 75% pada data uji (60 data dari total 300). Faktor yang paling berpengaruh terhadap prediksi kehadiran adalah jarak tempat tinggal ke lokasi event (47.73%), diikuti oleh usia peserta (36.00%).

Fitur-fitur yang digunakan dalam model:
- Jenis Kelamin
- Usia
- Jarak dari Rumah ke Lokasi Event (km)
- Status Pendaftaran
- Event ID (Event 1 atau Event 2)
- Riwayat Hadir di Event Sebelumnya

---

## Cara Menjalankan Secara Lokal

Pastikan Python 3.11 sudah terinstall, lalu jalankan perintah berikut dari folder skripsi:

Install dependensi:
```
pip install -r requirements.txt
```

Jalankan aplikasi:
```
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada alamat http://localhost:8501.

---

## Struktur Folder

```
skripsi/
├── .streamlit/
│   └── config.toml          # Konfigurasi Streamlit
├── data/
│   ├── raw/                  # Dataset mentah Excel (event1 dan event2)
│   └── processed/
│       └── dataset_gabungan.csv
├── models/
│   ├── random_forest_model.pkl
│   └── split_info.json
├── outputs/
│   ├── metrics_summary.json
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── distribusi_kehadiran.png
│   ├── distribusi_usia.png
│   └── distribusi_jarak.png
├── scripts/
│   ├── 01_preprocess.py
│   ├── 02_train.py
│   └── 03_evaluate.py
├── app.py
├── logo.png
├── requirements.txt
└── README.md
```

---

## Menjalankan Pipeline dari Awal

Jika ingin melatih ulang model dari data mentah, jalankan tiga script berikut secara berurutan:

```
python scripts/01_preprocess.py
python scripts/02_train.py
python scripts/03_evaluate.py
```

Script pertama membersihkan dan menggabungkan data dari dua file Excel mentah, script kedua melatih model Random Forest, dan script ketiga menghasilkan metrik evaluasi dan visualisasi yang digunakan oleh aplikasi.

---

## Dependensi

- streamlit untuk web app
- scikit-learn untuk model Random Forest
- pandas dan numpy untuk pengolahan data
- plotly untuk grafik interaktif
- matplotlib dan seaborn untuk grafik statis pada pipeline
- fpdf2 untuk ekspor laporan PDF
- pillow untuk pengolahan gambar
- openpyxl untuk membaca file Excel
