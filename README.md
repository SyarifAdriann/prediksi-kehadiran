# Prediksi Kehadiran Anggota Pelatihan
## PT Cahaya Ladara Nusantara — Algoritma Random Forest

Sistem prediksi kehadiran anggota pada kegiatan pelatihan makanan viral Sushi & Onigiri
menggunakan algoritma **Random Forest Classifier**, disertai aplikasi Streamlit interaktif
untuk visualisasi dan prediksi.

---

## Deskripsi Proyek

| Item | Detail |
|------|--------|
| **Judul Skripsi** | Prediksi Kehadiran Anggota Menggunakan Algoritma Random Forest Berdasarkan Data Riwayat Pelatihan Makanan Viral Sushi dan Onigiri di PT Cahaya Ladara Nusantara |
| **Program Studi** | Sistem Informasi — Universitas Dirgantara Marsekal Suryadarma |
| **Metode** | Knowledge Discovery in Database (KDD) + Random Forest |
| **Event 1** | Minggu, 09 November 2025 |
| **Event 2** | Minggu, 16 November 2025 |
| **Total Dataset** | 300 baris (150 per event, setelah digabung) |
| **Target Variabel** | Status Kehadiran (Hadir / Tidak Hadir) |

---

## Struktur Folder

```
skripsi/
├── data/
│   ├── raw/
│   │   ├── event1_raw.xlsx      # Dataset Event 1 (09 Nov 2025) — 150 baris
│   │   └── event2_raw.xlsx      # Dataset Event 2 (16 Nov 2025) — 150 baris
│   └── processed/
│       └── dataset_gabungan.csv # Dataset gabungan setelah preprocessing
├── models/
│   ├── random_forest_model.pkl  # Model Random Forest terlatih
│   └── split_info.json          # Metadata train/test split
├── outputs/
│   ├── metrics_summary.json     # Hasil evaluasi model (accuracy, precision, dll.)
│   ├── confusion_matrix.png     # Visualisasi confusion matrix
│   ├── feature_importance.png   # Visualisasi feature importance
│   ├── distribusi_kehadiran.png # Distribusi status kehadiran per event
│   ├── distribusi_usia.png      # Distribusi usia peserta
│   └── distribusi_jarak.png     # Distribusi jarak tempat tinggal
├── scripts/
│   ├── 01_preprocess.py         # Data selection, cleaning, merge, feature engineering
│   ├── 02_train.py              # Training model Random Forest
│   └── 03_evaluate.py           # Evaluasi model dan generate visualisasi
├── app.py                       # Aplikasi Streamlit
├── requirements.txt             # Daftar dependensi Python (pip)
├── environment.yml              # File environment Conda
├── README.md                    # Dokumentasi ini
└── tambahanskripsi.md           # Panduan penulisan tambahan skripsi
```

---

## ⚡ Quick Start (Fresh Clone)

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Letakkan file data di folder data/raw/
#    → data/raw/event1_raw.xlsx
#    → data/raw/event2_raw.xlsx

# 3. Jalankan seluruh pipeline
python scripts/01_preprocess.py && python scripts/02_train.py && python scripts/03_evaluate.py

# 4. Jalankan aplikasi
streamlit run app.py
```

---

## Cara Setup Environment

> **Prasyarat:** Python **3.11.x** diperlukan.
> Cek versi: `python --version`
> Download Python 3.11: https://www.python.org/downloads/release/python-3110/

### Opsi 1 — pip (direkomendasikan untuk pengguna biasa)

```bash
pip install -r requirements.txt
```

### Opsi 2 — Conda

```bash
conda env create -f environment.yml
conda activate skripsi-cln
```

---

## Prasyarat Data

> **⚠️ Penting:** File data mentah **tidak disertakan** dalam repositori ini.

Sebelum menjalankan pipeline, letakkan file berikut di folder `data/raw/`:

| File | Keterangan |
|------|------------|
| `event1_raw.xlsx` | Dataset Event 1 — Minggu, 09 November 2025 (150 baris) |
| `event2_raw.xlsx` | Dataset Event 2 — Minggu, 16 November 2025 (150 baris) |

Struktur kolom yang diperlukan: `Nama`, `Jenis_Kelamin`, `Usia`, `Jarak`, `Status_Pendaftaran`, `Status_Kehadiran`.

---

## Cara Menjalankan Pipeline

> **Penting:** Semua perintah dijalankan dari **root folder `skripsi/`**.

### Langkah 1 — Preprocessing

```bash
python scripts/01_preprocess.py
```

Melakukan: loading dataset, cleaning, standardisasi, parsing Jarak, merge, feature engineering, encoding, simpan ke `data/processed/dataset_gabungan.csv`.

### Langkah 2 — Training Model

```bash
python scripts/02_train.py
```

Melakukan: load dataset, split 80/20 stratified, training Random Forest, simpan model ke `models/random_forest_model.pkl`.

### Langkah 3 — Evaluasi & Visualisasi

```bash
python scripts/03_evaluate.py
```

Melakukan: load model, prediksi test set, hitung metrik (accuracy, precision, recall, F1), generate 5 chart, simpan ke `outputs/metrics_summary.json` dan `outputs/*.png`.

### (Opsional) Menjalankan Seluruh Pipeline Sekaligus

```bash
python scripts/01_preprocess.py && python scripts/02_train.py && python scripts/03_evaluate.py
```

---

## Cara Menjalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser secara otomatis di `http://localhost:8501`.

### Fitur Aplikasi

| Tab | Konten |
|-----|--------|
| **📊 Dashboard** | Ringkasan dataset, 5 chart interaktif (distribusi kehadiran, gender, usia, jarak, per event) |
| **🔮 Prediksi Kehadiran** | Form input manual → prediksi Hadir/Tidak Hadir + probabilitas + download CSV |
| **📈 Evaluasi Model** | Tabel metrik, confusion matrix, feature importance, download laporan PDF |

---

## Fitur-Fitur Model

| Fitur | Tipe | Keterangan |
|-------|------|------------|
| `Jenis_Kelamin` | Binary | 0=Perempuan, 1=Laki-Laki |
| `Usia` | Numerik | Usia peserta dalam tahun |
| `Jarak_km` | Numerik | Jarak tempat tinggal ke lokasi event (km) |
| `Status_Pendaftaran` | Binary | 1=Terdaftar, 0=Tidak Terdaftar |
| `Event_ID` | Kategorik | 1=Event 09 Nov, 2=Event 16 Nov |
| `hadir_event_sebelumnya` | Binary | 1=Hadir di event sebelumnya, 0=Tidak/N/A |

**Target:** `Status_Kehadiran` — 1=Hadir, 0=Tidak Hadir

---

## Hasil Model

| Metrik | Nilai |
|--------|-------|
| **Akurasi** | 75.00% |
| **Presisi (Weighted)** | 69.60% |
| **Recall (Weighted)** | 75.00% |
| **F1-Score (Weighted)** | 71.26% |

**Feature Importance (Urutan):**
1. Jarak (km) — 47.73%
2. Usia — 36.00%
3. Status Pendaftaran — 6.51%
4. Event ID — 4.22%
5. Hadir Event Sebelumnya — 3.35%
6. Jenis Kelamin — 2.18%

---

## Dependensi Utama

- `scikit-learn` — Random Forest Classifier
- `pandas` / `numpy` — Data processing
- `streamlit` — Web application
- `plotly` — Interactive charts
- `matplotlib` / `seaborn` — Static charts (pipeline)
- `fpdf2` — PDF report generation
- `openpyxl` — Baca file Excel

---

## Catatan Penting

1. **Data mentah harus disediakan terlebih dahulu** — lihat bagian [Prasyarat Data](#prasyarat-data) di atas sebelum menjalankan pipeline.
2. **Jalankan script secara berurutan**: `01` → `02` → `03` → `app.py`.
   Aplikasi Streamlit membutuhkan `models/random_forest_model.pkl` dan file `outputs/*.png` yang dihasilkan oleh pipeline.
3. **Gunakan Python 3.11.x** — Python 3.12+ belum diuji dan mungkin menyebabkan inkompatibilitas.
4. Seluruh script menggunakan `random_state=42` untuk reproduksibilitas.
5. Model menggunakan `class_weight='balanced'` untuk menangani distribusi kelas yang tidak seimbang (~79% Hadir).
6. File `tambahanskripsi.md` berisi panduan lengkap untuk penulisan BAB III, IV, dan V skripsi.
