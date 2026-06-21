# Setup Guide — Sistem Prediksi Kehadiran Anggota
## PT Cahaya Ladara Nusantara

> **Untuk Claude Code:** Baca file ini dari awal. Periksa environment laptop terlebih dahulu, pasang yang belum ada, lalu jalankan pipeline secara berurutan hingga Streamlit berjalan. Semua perintah dijalankan dari folder root proyek ini (`skripsi/`).

---

## 1. Overview Proyek

Sistem prediksi kehadiran anggota berbasis **Random Forest** (scikit-learn). Dibangun sebagai skripsi, dilengkapi aplikasi web Streamlit.

**Alur kerja penuh:**
```
Raw Excel → 00_anonymize.py → 01_preprocess.py → 02_train.py → 03_evaluate.py → streamlit run app.py
```

**Struktur folder penting:**
```
skripsi/
├── app.py                          ← Streamlit app (entry point)
├── logo.png                        ← Logo PT CLN (dipakai di app)
├── requirements.txt                ← Semua dependensi Python
├── environment.yml                 ← Alternatif: Conda environment
├── data/
│   ├── raw/
│   │   ├── event1_raw.xlsx         ← Data mentah Event 1 (sudah dianonimkan)
│   │   └── event2_raw.xlsx         ← Data mentah Event 2 (sudah dianonimkan)
│   ├── processed/
│   │   ├── dataset_gabungan.csv    ← Output preprocessing (input ke model)
│   │   └── anon_mapping.csv        ← Tabel mapping kode → nama (simpan aman)
│   └── raw_backup_original/        ← Backup data asli sebelum anonimisasi
├── models/
│   ├── random_forest_model.pkl     ← Model terlatih
│   ├── split_info.json             ← Info train/test split
│   └── best_params.json            ← Hasil GridSearchCV
├── outputs/
│   ├── metrics_summary.json        ← Metrik evaluasi model
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── distribusi_kehadiran.png
│   ├── distribusi_usia.png
│   └── distribusi_jarak.png
├── scripts/
│   ├── 00_anonymize.py             ← Anonimisasi nama (sudah dijalankan)
│   ├── 01_preprocess.py            ← KDD: Data cleaning & feature engineering
│   ├── 02_train.py                 ← Training Random Forest + GridSearchCV
│   └── 03_evaluate.py              ← Evaluasi & generate charts
└── .streamlit/
    └── config.toml                 ← Konfigurasi Streamlit
```

---

## 2. Persyaratan Sistem

### Python
- **Versi wajib: Python 3.11.x**
- Python 3.12+ belum tentu kompatibel (scikit-learn & beberapa lib masih dioptimalkan untuk 3.11)
- Cek versi: `python --version`

### Cara install Python 3.11 (jika belum ada)
- Windows: https://www.python.org/downloads/release/python-3119/
- Pilih "Windows installer (64-bit)"
- Centang "Add Python to PATH" saat instalasi

### pip
- Biasanya sudah termasuk bersama Python
- Cek: `pip --version`
- Update jika perlu: `python -m pip install --upgrade pip`

---

## 3. Cara Setup (Pilih salah satu)

### Opsi A — pip + venv (Direkomendasikan untuk laptop baru)

```bash
# 1. Buat virtual environment
python -m venv venv

# 2. Aktifkan (Windows)
venv\Scripts\activate

# 3. Install semua dependensi
pip install -r requirements.txt

# 4. Verifikasi
python -c "import sklearn, streamlit, pandas, joblib, plotly, fpdf; print('OK')"
```

### Opsi B — Conda / Miniconda

```bash
# 1. Buat environment dari file
conda env create -f environment.yml

# 2. Aktifkan
conda activate skripsi-cln

# 3. Install fpdf2 via pip (tidak tersedia di conda-forge)
pip install fpdf2>=2.7.0

# 4. Verifikasi
python -c "import sklearn, streamlit, pandas, joblib, plotly, fpdf; print('OK')"
```

---

## 4. Daftar Dependensi Lengkap

| Library | Versi Minimum | Fungsi |
|---|---|---|
| `pandas` | 2.0.0 | Manipulasi dataset (DataFrame) |
| `numpy` | 1.24.0 | Komputasi numerik |
| `openpyxl` | 3.1.0 | Baca file `.xlsx` |
| `scikit-learn` | 1.3.0 | Random Forest, GridSearchCV, metrik evaluasi |
| `joblib` | 1.3.0 | Simpan/load model `.pkl` |
| `matplotlib` | 3.7.0 | Generate chart PNG (pipeline) |
| `seaborn` | 0.12.0 | Heatmap confusion matrix |
| `plotly` | 5.15.0 | Chart interaktif di Streamlit |
| `streamlit` | 1.28.0 | Web app framework |
| `pillow` | 9.0.0 | Baca file gambar (logo) |
| `fpdf2` | 2.7.0 | Export laporan PDF dari app |

---

## 5. Menjalankan Pipeline (jika perlu retrain dari awal)

> **Catatan:** Jika folder `models/` dan `outputs/` sudah ada dan berisi file `.pkl` dan `.json`, langsung loncat ke **Step 6 (jalankan app)**. Tidak perlu retrain kecuali ada data baru.

### Step 0 — Anonimisasi (SKIP jika data sudah dianonimkan)

Data di `data/raw/` sudah dianonimkan. Jalankan ini hanya jika menggunakan data mentah baru yang masih berisi nama asli:

```bash
python scripts/00_anonymize.py
```

Output: `data/processed/anon_mapping.csv` (simpan aman, jangan dipublikasikan)

### Step 1 — Preprocessing

```bash
python scripts/01_preprocess.py
```

Output: `data/processed/dataset_gabungan.csv`

Yang dilakukan:
- Baca `event1_raw.xlsx` & `event2_raw.xlsx`
- Bersihkan data (standardisasi kecamatan, parse jarak, koreksi outlier)
- Hitung fitur `hadir_event_sebelumnya` (matching peserta lintas event berdasarkan ID)
- Encode fitur kategorik
- Simpan CSV gabungan 300 baris

### Step 2 — Training Model

```bash
python scripts/02_train.py
```

Output: `models/random_forest_model.pkl`, `models/split_info.json`, `models/best_params.json`

Yang dilakukan:
- Train/test split 80/20 (stratified, random_state=42)
- GridSearchCV: 108 kombinasi hyperparameter × 5-fold CV
- **Estimasi waktu: 2–5 menit** (tergantung spek laptop)
- Best params yang diharapkan: `n_estimators=100, max_depth=10, min_samples_split=5`
- Target accuracy: ~83%

### Step 3 — Evaluasi & Visualisasi

```bash
python scripts/03_evaluate.py
```

Output: `outputs/metrics_summary.json` + 5 file chart PNG

Yang dilakukan:
- Rekonstruksi test set yang sama dengan training
- Hitung accuracy, precision, recall, F1
- Generate confusion matrix, feature importance, distribusi dataset

---

## 6. Menjalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

Buka browser: **http://localhost:8501**

**Login default:**
- Username: `admin`
- Password: `supervisor`

> Login bisa diubah dengan membuat file `.streamlit/secrets.toml`:
> ```toml
> [credentials]
> username = "admin"
> password = "passwordbaru"
> ```

### Tab-tab di aplikasi:
| Tab | Fungsi |
|---|---|
| **Prediksi Kehadiran** | Input data peserta baru → prediksi hadir/tidak hadir |
| **Dashboard** | Visualisasi interaktif distribusi dataset |
| **Evaluasi Model** | Confusion matrix, feature importance, metrik |
| **Log Prediksi** | Riwayat semua prediksi yang sudah dibuat |

---

## 7. File yang Dibutuhkan Agar App Berjalan

App akan error jika file-file ini tidak ada. Pastikan semuanya ada sebelum `streamlit run`:

| File | Dibutuhkan oleh | Cara generate jika hilang |
|---|---|---|
| `data/processed/dataset_gabungan.csv` | Dashboard, Log tab | Jalankan `01_preprocess.py` |
| `models/random_forest_model.pkl` | Tab Prediksi, semua fitur prediksi | Jalankan `02_train.py` |
| `models/split_info.json` | `03_evaluate.py` | Jalankan `02_train.py` |
| `outputs/metrics_summary.json` | Sidebar (akurasi), Tab Evaluasi | Jalankan `03_evaluate.py` |
| `outputs/confusion_matrix.png` | Tab Evaluasi | Jalankan `03_evaluate.py` |
| `outputs/feature_importance.png` | Tab Evaluasi | Jalankan `03_evaluate.py` |
| `logo.png` | Sidebar & login page | Sudah ada di repo, jangan dihapus |

---

## 8. Troubleshooting

### `ModuleNotFoundError: No module named 'X'`
```bash
pip install -r requirements.txt
```

### Streamlit tidak terbuka di browser
```bash
streamlit run app.py --server.port 8502
```
Lalu buka http://localhost:8502

### `UnicodeEncodeError` saat menjalankan script Python di Windows
```bash
$env:PYTHONUTF8=1; python scripts/01_preprocess.py
```
Tambahkan `$env:PYTHONUTF8=1;` sebelum semua perintah python.

### Training sangat lama (> 10 menit)
GridSearchCV menggunakan `n_jobs=-1` (semua core CPU). Jika tetap lambat, edit `02_train.py` dan kurangi `param_grid`:
```python
param_grid = {
    "n_estimators":      [100, 200],      # kurangi dari [100, 200, 300]
    "max_depth":         [None, 10],      # kurangi dari [None, 10, 20, 30]
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
}
```

### Error `FileNotFoundError` untuk file Excel
Pastikan file ada di `data/raw/`:
```
data/raw/event1_raw.xlsx
data/raw/event2_raw.xlsx
```

---

## 9. Catatan Penting

- **Random seed = 42** digunakan di seluruh pipeline. Jangan ubah agar hasil reprodusibel.
- **`anon_mapping.csv`** berisi mapping kode → nama asli peserta. Jangan dipublikasikan atau diupload ke repo publik.
- **`data/raw_backup_original/`** berisi data asli sebelum anonimisasi. Simpan aman.
- App tidak memerlukan internet untuk berjalan (semua lokal), kecuali font Inter dari Google Fonts di browser (kosmetik saja, tidak mempengaruhi fungsi).
