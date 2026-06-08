# Extract 02 — Metrics, Data Samples & Existing Documentation

**Project:** Sistem Prediksi Kehadiran Anggota PT Cahaya Ladara Nusantara  
**Generated:** 2026-06-07

---

## SECTION 1 — `outputs/metrics_summary.json` (Complete)

```json
{
  "n_test": 60,
  "n_train": 240,
  "n_total": 300,
  "accuracy": 0.75,
  "accuracy_pct": 75.0,
  "precision_weighted": 0.696,
  "recall_weighted": 0.75,
  "f1_weighted": 0.7126,
  "per_class": {
    "Hadir": {
      "precision": 0.7963,
      "recall": 0.9149,
      "f1": 0.8515,
      "support": 47
    },
    "Tidak Hadir": {
      "precision": 0.3333,
      "recall": 0.1538,
      "f1": 0.2105,
      "support": 13
    }
  },
  "confusion_matrix": {
    "TN": 2,
    "FP": 11,
    "FN": 4,
    "TP": 43
  },
  "feature_importance": {
    "Jenis_Kelamin": 0.02184692662759057,
    "Usia": 0.36001692262266394,
    "Jarak_km": 0.4772506391036618,
    "Status_Pendaftaran": 0.06510348744053576,
    "Event_ID": 0.04224555075810258,
    "hadir_event_sebelumnya": 0.033536473447445495
  },
  "feature_importance_sorted": [
    { "feature": "Jarak_km", "importance": 0.4773 },
    { "feature": "Usia", "importance": 0.36 },
    { "feature": "Status_Pendaftaran", "importance": 0.0651 },
    { "feature": "Event_ID", "importance": 0.0422 },
    { "feature": "hadir_event_sebelumnya", "importance": 0.0335 },
    { "feature": "Jenis_Kelamin", "importance": 0.0218 }
  ],
  "model_params": {
    "n_estimators": 100,
    "random_state": 42,
    "class_weight": "balanced",
    "test_size": 0.2
  },
  "distribusi_dataset": {
    "total": 300,
    "hadir": 236,
    "tidak_hadir": 64,
    "hadir_pct": 78.67
  }
}
```

---

## SECTION 2 — `models/split_info.json` (Complete)

```json
{
  "test_size": 0.2,
  "random_seed": 42,
  "feature_cols": [
    "Jenis_Kelamin",
    "Usia",
    "Jarak_km",
    "Status_Pendaftaran",
    "Event_ID",
    "hadir_event_sebelumnya"
  ],
  "target_col": "Status_Kehadiran",
  "n_train": 240,
  "n_test": 60,
  "train_hadir": 189,
  "train_tidak_hadir": 51,
  "test_hadir": 47,
  "test_tidak_hadir": 13
}
```

---

## SECTION 3 — `data/processed/dataset_gabungan.csv` (First 10 Rows)

**Total rows:** 300  
**Total columns:** 17  

**Column names:**
`ID ANGGOTA`, `Jenis Kelamin`, `Status Pendaftaran`, `Status Kehadiran`, `Jarak`, `Kecamatan`, `Kabupaten/Kota`, `Provinsi`, `Desa/Kelurahan`, `Alamat tempat tinggal`, `Jenis_Kelamin`, `Usia`, `Jarak_km`, `Status_Pendaftaran`, `Event_ID`, `hadir_event_sebelumnya`, `Status_Kehadiran`

**First 10 rows (CSV format):**

```csv
"ID ANGGOTA","Jenis Kelamin","Status Pendaftaran","Status Kehadiran","Jarak","Kecamatan","Kabupaten/Kota","Provinsi","Desa/Kelurahan","Alamat tempat tinggal","Jenis_Kelamin","Usia","Jarak_km","Status_Pendaftaran","Event_ID","hadir_event_sebelumnya","Status_Kehadiran"
"Rosmiyanti","Perempuan","Terdaftar","Hadir","12 KM","Setiabudi","Jakarta Selatan","DKI Jakarta","Pasar Manggis","Rusunawa Pasar Rumput, Setiabudi Jaksel","0","37","12.0","1","1","0","1"
"Agusnawati","Perempuan","Terdaftar","Hadir","2.5 KM","Kramat Jati","Jakarta Timur","DKI Jakarta","Kelurahan Cililitan","Jl. Ciliwung no. 114 rt 11 rw 06","0","45","2.5","1","1","0","1"
"Dra.Endaruna","Perempuan","Terdaftar","Tidak Hadir","22 KM","Cakung","Jakarta Timur","DKI Jakarta","Kelurahan Pulo Gebang","Jl Garuda VII  Perumahan Pulo Gebang Permai Blok G6 No 5","0","50","22.0","1","1","0","0"
"Soraya safitri","Perempuan","Terdaftar","Hadir","2 KM","Kramat Jati","Jakarta Timur","DKI Jakarta","Cililitan","Jl Dewi Sartika no 10 rt 005 rw 013","0","52","2.0","1","1","0","1"
"Asep Sudjana","Laki - Laki","Terdaftar","Hadir","9.5 KM","Pulo Gadung","Jakarta Timur","DKI Jakarta","Pisangan Timur","Jl Pisangan lama III no 3 RT03 RW07 Pisangan Timur Pulo gadung","1","46","9.5","1","1","0","1"
"Chairunnisyah Nst","Perempuan","Terdaftar","Tidak Hadir","12 KM","Matraman","Jakarta Timur","DKI Jakarta","Kelurahan Kayu Manis","Jl. Kayu Manis Barat Gg. K2 No.15 Rt 019/04","0","27","12.0","1","1","0","0"
"Shanty Puspa oktavianty","Perempuan","Terdaftar","Hadir","400 M","Makasar","Jakarta Timur","DKI Jakarta","Kebon pala","Jl Kamboja gg Cempaka no 6A RT 10 RW 01","0","28","0.4","1","1","0","1"
"Sri hartaty","Perempuan","Terdaftar","Tidak Hadir","13 KM","Tanjung Priok","Jakarta Utara","DKI Jakarta","Sunter Jaya","Jl Telaga Permata 6 Rt009/001 No. 18A","0","52","13.0","1","1","0","0"
"Ambar Kriswijayanti","Perempuan","Terdaftar","Hadir","26 KM","Cilincing","Jakarta Utara","DKI Jakarta","Sukapura","Jl  Pelajar No. 23","0","49","26.0","1","1","0","1"
"Jl kp Pulo nrt 6 RW 5 pinang Ranti","Perempuan","Terdaftar","Hadir","1.2 KM","Makasar","Jakarta Timur","DKI Jakarta","Makasar","Jl kerja bakti IV no 13B","0","36","1.2","1","1","0","1"
```

---

## SECTION 4 — `results_defense_master.md` (Complete)

# Thesis Defense Audit & Results Master

**Project Status:** High-Risk  
**Focus:** Repository Evidence, Quantitative Validation, Methodological Scrutiny

This document acts as an uncompromising defense audit for the Attendance Prediction System. It is designed to prepare the candidate for extreme scrutiny by a knowledgeable examiner.

---

### 1. Locate All Result Sources

Every result, metric, and visualization in this thesis is generated programmatically and stored in the repository. There is no manual data entry for results.

*   **`scripts/03_evaluate.py`**: The single source of truth that generates all metrics and images.
*   **`outputs/metrics_summary.json`**: Contains exact numerical results of the final model (Accuracy, Precision, Recall, F1, Confusion Matrix, Feature Importance).
*   **`outputs/confusion_matrix.png`**: Heatmap of the TP/TN/FP/FN.
*   **`outputs/feature_importance.png`**: Bar chart of the 6 features.
*   **`outputs/distribusi_*.png`**: Charts showing distributions of attendance, age, and distance.
*   **`models/random_forest_model.pkl`**: The actual trained model artifact.
*   **`models/split_info.json`**: Metadata confirming the exact test set size (60 rows) and class split.
*   **`app.py` (Tab "Evaluasi Model")**: Dynamically loads `metrics_summary.json` and images to display to end-users.
*   **`pred_log.json`**: Stores real-time predictions generated on localhost.

---

### 2. Extract Actual Model Performance

Based entirely on `outputs/metrics_summary.json` (Test Set Size = 60):

*   **Final Thesis Model:** Random Forest Classifier (XGBoost and SMOTE iterations were deleted from the repository).
*   **Accuracy:** 75.0%
*   **Precision (Weighted):** 69.6%
*   **Recall (Weighted):** 75.0%
*   **F1-score (Weighted):** 71.26%

*Note: The metrics report "Weighted" averages, which mask poor performance on the minority class. We must look at the per-class metrics to see the reality.*

---

### 3. Confusion Matrix Deep Audit

Based on `metrics_summary.json`:

*   **True Positives (TP):** 43 (Actual Hadir, Predicted Hadir)
*   **True Negatives (TN):** 2 (Actual Tidak Hadir, Predicted Tidak Hadir)
*   **False Positives (FP):** 11 (Actual Tidak Hadir, Predicted Hadir)
*   **False Negatives (FN):** 4 (Actual Hadir, Predicted Tidak Hadir)

**Mistake Profile:**
The model heavily overpredicts "Hadir". It missed 11 out of 13 absent people, guessing they would attend.

**Calculations:**
*   **Sensitivity / Recall (Hadir):** 43 / (43 + 4) = 91.49%
*   **Specificity / Recall (Tidak Hadir):** 2 / (2 + 11) = 15.38%
*   **False Positive Rate:** 11 / 13 = 84.62%
*   **False Negative Rate:** 4 / 47 = 8.51%

---

### 4. Majority-Class Baseline Test (Brutally Honest)

*   Total Dataset: 300 rows
*   Hadir Count: 236 (78.67%)
*   Tidak Hadir Count: 64 (21.33%)

**Baseline "Guess Hadir" Accuracy: 78.67%**  
**Random Forest Accuracy: 75.00%**

**Verdict:** The Random Forest model **loses to the naive baseline by 3.67%**.

---

### 5. Feature Importance Audit

| Rank | Feature | Importance | Business Logic |
| :--- | :--- | :--- | :--- |
| 1 | `Jarak_km` | 47.73% | Logical. Commute friction heavily dictates physical attendance. |
| 2 | `Usia` | 36.00% | Logical. Age cohorts have different availabilities/health. |
| 3 | `Status_Pendaftaran` | 6.51% | Weak. Registration intent should logically be #1 or #2. |
| 4 | `Event_ID` | 4.22% | Indicates slight variance between the Nov 09 and Nov 16 events. |
| 5 | `hadir_event_sebelumnya` | 3.35% | Surprisingly low. Loyalty/history is usually a top predictor. |
| 6 | `Jenis_Kelamin` | 2.18% | Logical. Gender rarely dictates attendance in general events. |

---

### 6. Prediction Distribution Audit

*   Actual Test Set: 47 Hadir (78%), 13 Tidak Hadir (22%)
*   Model Predictions: 54 Hadir (90%), 6 Tidak Hadir (10%)

**Class Bias:** The model is highly biased toward the majority class.

---

### 7. Misclassification Analysis

**Common Pattern for FP (Predicted Hadir, Actual Tidak Hadir):**
Because `Jarak_km` and `Usia` control 83% of the decision, if a participant lives close by (e.g., < 5km) and is in the average age bracket, the model is almost guaranteed to predict "Hadir".

---

### 8. Leakage Investigation (CRITICAL)

**IID Violation:**
If Participant "Budi" is in Event 1 and Event 2, his Event 1 row might go to Train, and Event 2 row to Test. The model learns Budi's exact `Jarak_km` and `Usia` in Train, and predicts on the same `Jarak_km` and `Usia` in Test.

**Time Travel Leakage:**
Because Event 1 (Past) and Event 2 (Future) are shuffled together, the model is trained on Future data to predict Past data.

**Leakage Risk:** **HIGH**

---

### 9. Robustness Audit

*   **Sample Size:** 300 rows total. 60 rows for testing. A change of 1 prediction alters accuracy by 1.6%.
*   **Imbalance:** 78% vs 21%. The model heavily overfits the majority class.
*   **Engineered Features:** `Jarak_km` adds random noise `np.random.uniform(-0.5, 1.5)`. Breaks determinism.

---

### 10. Examiner Attack Scenarios

**Q1: Why is your model's accuracy (75%) worse than guessing "Hadir" for everyone (78%)?**
*   *Strong Defense:* "Tebakan mayoritas memiliki Specificity 0%. Model ini mengorbankan sedikit akurasi keseluruhan demi mendapatkan Specificity 15%, yaitu kemampuan mendeteksi sebagian orang yang akan absen."

**Q2: Why did you use random `train_test_split` on time-ordered data?**
*   *Strong Defense:* "Ini adalah batasan penelitian ini. Karena total data hanya 300, memotong berdasarkan waktu akan membuat data latih terlalu sedikit (150 baris). Saya menyadari risiko IID violation ini."

**Q3: `Jarak_km` mendominasi 47% tapi dikoreksi menggunakan random noise?**
*   *Strong Defense:* "Noise tersebut sangat kecil (-0.5 sampai 1.5 km) dan hanya pada outlier. Tujuannya mencegah model membuat over-split pada satu titik nilai eksak centroid."

**Q4: Anda bilang pakai SMOTE dan XGBoost di proposal, mana di sistem akhir?**
*   *Strong Defense:* "Setelah eksperimen, XGBoost dan SMOTE dibuang karena Random Forest standar dengan `class_weight='balanced'` memberikan kompromi terbaik."

**Q5: Nilai F1-Score kelas Tidak Hadir hanya 21%. Apakah model ini bisa diandalkan?**
*   *Strong Defense:* "Belum cukup handal untuk produksi skala besar. Model ini adalah proof-of-concept yang butuh lebih banyak data absensi."

---

### 11. Final Verdict

*   **Technical Quality:** 8/10
*   **Methodological Quality:** 3/10
*   **Defense Readiness:** 4/10

**Classification:** **HIGH RISK**

**Recommendation:** The student MUST NOT brag about accuracy. Focus on "Membangun Sistem (Engineering)" rather than "Kecerdasan Model (Data Science)". Acknowledge ML flaws proactively as *Saran untuk penelitian selanjutnya*.

---

## SECTION 5 — `training_pipeline_master.md` (Complete)

# Master Training Pipeline Trace

This document provides an exhaustive, end-to-end trace of the machine learning pipeline implemented in this repository, mapping a single record from raw Excel input to final model prediction, alongside an analysis of methodological decisions.

---

### 1. Raw Source Files

The pipeline begins with two Excel files:
*   **File 1:** `data/raw/event1_raw.xlsx`
*   **File 2:** `data/raw/event2_raw.xlsx`
*   **Total Row Count:** 300 rows combined.
*   **Raw Columns (12):** `"No.", "ID ANGGOTA", "Usia", "Jenis Kelamin", "Alamat tempat tinggal", "Desa/Kelurahan", "Kecamatan", "Kabupaten/Kota", "Provinsi", "Jarak", "Status Pendaftaran", "Status Kehadiran"`

---

### 2. Preprocessing Walkthrough (Line-by-Line)

Executed in `scripts/01_preprocess.py`. The `No.` column is dropped immediately.

**Transformation 1: String Standardization**
*   Code Location: `standardize_kecamatan(val)`, `standardize_kab(val)`
*   Original Value: `" Jakarta Utata "`
*   Transformed Value: `"Jakarta Utara"`
*   Reason: Fixes typos, strips whitespaces, standardizes casing.

**Transformation 2: Missing Value Handling (Categorical)**
*   Code Location: `df[col] = df[col].replace("nan", None)`
*   Original Value: `"nan"` (string representation of pandas NaN)
*   Transformed Value: `None`
*   Reason: Ensures missing categorical values are treated as actual nulls.

**Transformation 3: Distance Parsing (`parse_jarak`)**
*   Code Location: `parse_jarak(val)`
*   Original Value: `"13,8 KM"` or `"400 M"`
*   Transformed Value: `13.8` or `0.4`
*   Reason: Converts inconsistent human-entered text strings into standard float km values.

---

### 3. Deep Dive: `hadir_event_sebelumnya`

*   **Implementation Location:** `engineer_hadir_event_sebelumnya(df)` in `01_preprocess.py`
*   **Lookup Logic:**
    1. Isolates all rows belonging to Event 1.
    2. Creates a dictionary mapping lowercased, stripped `ID ANGGOTA` to their `Status Kehadiran`.
*   **Matching Logic:**
    *   Event 1 rows: forced to `0` (no preceding event).
    *   Event 2 rows: looks up name in Event 1 dictionary. Found AND "Hadir" → value = `1`. Otherwise `0`.
*   **Example:** Budi attends Event 1 (Hadir). Budi registers for Event 2. When processing his Event 2 row, "budi" is found in the lookup with status "Hadir". `hadir_event_sebelumnya = 1`.

---

### 4. Deep Dive: `Jarak_km` (Haversine & Outliers)

*   **Implementation Location:** `correct_jarak_outliers(df)` in `01_preprocess.py`
*   **Haversine Calculation:** Calculates exact spherical distance between event coordinates (Halim: -6.2634, 106.8910) and hardcoded centroid coordinates of participant's `Kecamatan`.
*   **Outlier Detection Rules:**
    1. Input Distance > `3 * Centroid Distance` AND
    2. Input Distance > `Centroid Distance + 10 km`
*   **Replacement:** If flagged: `Centroid Distance + np.random.uniform(-0.5, 1.5)`. Slight variance mimics realistic household distributions. Missing values filled with dataset median.

---

### 5. Train/Test Split Details

*   **Total Dataset:** 300 rows
*   **Split Ratio:** 80% Train / 20% Test (stratified by `y`)
*   **Training Set:** 240 rows — Hadir: 189, Tidak Hadir: 51
*   **Test Set:** 60 rows — Hadir: 47, Tidak Hadir: 13

---

### 6. Single Record Trace (End-to-End)

1. **Raw Row:** Jenis Kelamin: "Perempuan", Usia: 25, Kecamatan: "Makassar", Jarak: "50 KM", Event_ID: 2 (attended Event 1 previously)
2. **Preprocessing:** `Kecamatan` standardized to `"Makasar"`. `Jarak` parsed to `50.0`.
3. **Feature Engineering:** 50km for Makasar flagged as outlier (centroid ~2km). Replaced with `2.0 + 0.8 = 2.8 km`. Lookup confirms Event 1 attendance → `hadir_event_sebelumnya = 1`.
4. **Encoded Values:** "Perempuan" → `0`.
5. **Model Input Vector:** `[0, 25, 2.8, 1, 2, 1]`
6. **Prediction:** `predict_proba()` returns `[0.12, 0.88]`. `predict()` returns `1` (Hadir).

---

### 7. Potential Examiner Challenges

**1. Dataset Size is Too Small**
*   Criticism: 300 rows is very small for Random Forest. Test set only 60 rows; a change of 3 predictions shifts accuracy by 5%.
*   Defense: Valid. For a thesis, 300 rows is acceptable for manually collected primary data. `class_weight="balanced"` and `stratify=y` mitigate small-sample bias.

**2. Random Noise in Distance Imputation**
*   Criticism: Injecting `np.random.uniform(-0.5, 1.5)` into outlier distances alters deterministic spatial data.
*   Defense: Partially valid. Prevents Random Forest from creating artificially dense node at exact centroid values.

**3. Leakage in Event 1 `hadir_event_sebelumnya`**
*   Criticism: All Event 1 participants are hardcoded to `0`. So `0` means two different things: "First event" AND "Did not attend."
*   Defense: Valid. Better approach: use Event 1 only as historical data, train exclusively on Event 2. But with only 300 rows, discarding Event 1 halves the dataset. `Event_ID` feature allows tree to split logic by event.

**4. Overfitting via Hyperparameter Defaulting**
*   Criticism: Hardcoded `n_estimators=100` without GridSearchCV.
*   Defense: Valid. Earlier `04_experiment.py` (now deleted) ran GridSearchCV. Optimal params were hardcoded into final `02_train.py`.

---

## SECTION 6 — `repo_learning_map.md` (Complete)

# Repository Reconnaissance Report: Attendance Prediction System

This document provides a comprehensive map of the thesis repository for educational purposes.

### 1. Repository Structure

**Folders:**
*   `data/` — raw/ and processed/. (Important)
*   `scripts/` — Core ML pipeline scripts. (Critical)
*   `models/` — Trained model and split metadata. (Critical)
*   `outputs/` — Evaluation metrics, charts, visualizations. (Supporting)
*   `figures/` — Diagrams and illustrations for the thesis. (Supporting)
*   `.streamlit/` — Configuration and secrets. (Supporting)

**Python Files:**
*   `scripts/01_preprocess.py` — Extracts, cleans, standardizes, engineers features, encodes. (Critical)
*   `scripts/02_train.py` — Splits data, trains Random Forest. (Critical)
*   `scripts/03_evaluate.py` — Generates metrics and visualizations. (Critical)
*   `app.py` — Streamlit web application. (Critical)

**Data Files:**
*   `data/raw/event1_raw.xlsx` — Raw attendance data Event 1. (Important)
*   `data/raw/event2_raw.xlsx` — Raw attendance data Event 2. (Important)
*   `data/processed/dataset_gabungan.csv` — Merged preprocessed dataset. (Critical)

**Model & Output Files:**
*   `models/random_forest_model.pkl` — Serialized Random Forest. (Critical)
*   `models/split_info.json` — Split metadata. (Supporting)
*   `outputs/metrics_summary.json` — Full evaluation metrics. (Supporting)
*   `outputs/*.png` — Evaluation charts. (Supporting)

---

### 2. Actual ML Pipeline Discovery

1. **Raw Data Source:** `event1_raw.xlsx` and `event2_raw.xlsx`.
2. **Preprocessing:** `load_event()` uses `pandas.read_excel()`. `standardize_kecamatan()` and `standardize_kab()` clean text.
3. **Feature Engineering:** `correct_jarak_outliers()` + `haversine_km()`. `engineer_hadir_event_sebelumnya()`. `encode_features()`.
4. **Train/Test Split:** `train_test_split` from sklearn. Variables: `X_train`, `X_test`, `y_train`, `y_test`.
5. **Model Training:** `RandomForestClassifier`. `model.fit(X_train, y_train)`.
6. **Evaluation:** `accuracy_score()`, `precision_score()`, `recall_score()`, `f1_score()`, `confusion_matrix()`. `model.feature_importances_`.
7. **Prediction:** `model.predict(X_input)` and `model.predict_proba(X_input)` in `app.py`.

---

### 3. Dataset Inventory

*   **Target Variable:** `Status_Kehadiran` (1 = Hadir, 0 = Tidak Hadir)
*   **Model Features:** `Jenis_Kelamin`, `Usia`, `Jarak_km`, `Status_Pendaftaran`, `Event_ID`, `hadir_event_sebelumnya`
*   **`hadir_event_sebelumnya`** created in `engineer_hadir_event_sebelumnya(df)` in `01_preprocess.py`. Lookup: name from Event 1 → if "Hadir" then 1, else 0. Event 1 rows always 0.

---

### 4. Feature Engineering Inventory

**1. Distance Parsing and Outlier Correction (`Jarak_km`)**
*   Source: `Jarak`, `Kecamatan` → Result: `Jarak_km`
*   `parse_jarak`: Regex extraction from strings. `correct_jarak_outliers`: Haversine to centroid, flag if > 3x and > +10km, replace with centroid + uniform noise.

**2. Historical Attendance (`hadir_event_sebelumnya`)**
*   Source: `Event_ID`, `ID ANGGOTA`, `Status Kehadiran` → Result: `hadir_event_sebelumnya`
*   Case-insensitive name matching.

**3. Categorical Encoding**
*   Source: `Jenis Kelamin`, `Status Pendaftaran`, `Status Kehadiran` → `Jenis_Kelamin`, `Status_Pendaftaran`, `Status_Kehadiran`
*   Dictionary mapping: "Perempuan" → 0, "Laki-Laki" → 1.

---

### 5. Random Forest Inventory

*   **File:** `scripts/02_train.py`
*   **n_estimators:** 100
*   **class_weight:** "balanced"
*   **random_state:** 42
*   **test_size:** 0.20
*   **stratify:** y

---

### 6. Evaluation Inventory

All in `scripts/03_evaluate.py`:

*   `accuracy_score(y_test, y_pred)` — Line 126
*   `precision_score(..., average="weighted")` — Line 127
*   `recall_score(..., average="weighted")` — Line 128
*   `f1_score(..., average="weighted")` — Line 129
*   Per-class: Lines 132–137
*   `confusion_matrix(y_test, y_pred)` — Line 139
*   `model.feature_importances_` — Line 155

---

### 7. Streamlit Inventory

*   **Entry Point:** `app.py`
*   **Security Gate:** `_check_login()` reads `st.secrets["credentials"]`. `st.stop()` blocks rendering.
*   **Tabs:** `tab_pred`, `tab_dash`, `tab_eval`, `tab_log`
*   **Prediction Flow:**
    1. User enters data via `st.selectbox` and `st.number_input`.
    2. Clicks "Prediksi Sekarang".
    3. Inputs encoded to integers.
    4. `X_input` DataFrame created.
    5. `model.predict(X_input)` generates class.
    6. `model.predict_proba(X_input)` generates probabilities.
    7. Stored in `st.session_state['pred_data']` and `pred_log`.
    8. `st.rerun()` refreshes UI.

---

### 8. Thesis Consistency Check

| Thesis Claim | Code Reality | Match? |
| :--- | :--- | :--- |
| System saves to Database | Saves to `st.session_state` and `pred_log.json` file. No database. | ❌ No |
| System "Cetak Laporan" | Uses `st.download_button` for CSV/PDF. No printing. | ❌ No |
| Uses XGBoost / SMOTE | Only `RandomForestClassifier`. XGBoost scripts deleted. | ❌ No |
| `hadir_event_sebelumnya` exists natively | Manually engineered via name-matching across two Excel files. | ❌ No |

---

### 9. Learning Difficulty Ranking

1. **Outlier Correction & Haversine Math** — `correct_jarak_outliers()`: Haversine formula, centroid mapping, 3x+10km boundary, random noise injection.
2. **Cross-Event Feature Engineering** — `engineer_hadir_event_sebelumnya()`: Lookup dictionaries, row-by-row iteration, edge cases.
3. **Streamlit Session State & Logic Gates** — Top-to-bottom reruns, `st.session_state` for login and log persistence.
4. **Evaluation Metrics** — Weighted vs per-class metrics, confusion matrix interpretation.
5. **Random Forest Training** — Standard sklearn boilerplate. Easiest section.

---

## SECTION 7 — `leftover.md` (Complete)

# Session Leftover Notes

## What we were doing
We just added a secure login gate to the Streamlit application to restrict access. This was Option 1 (Simple password gate) requested by you.

## How far along we got
1. **Credentials Setup:** Created `.streamlit/secrets.toml` locally with the requested credentials (`username = "admin"`, `password = "supervisor"`).
2. **Login UI:** Inserted a full-screen login gate in `app.py` that matches the app's premium visual design (navy/white, shadowed card, centered logo).
3. **Session Logic:** Implemented `st.session_state["logged_in"]` checks. `st.stop()` is used to prevent the rest of the app from rendering until login is successful.
4. **Logout Feature:** Added a "Keluar" (Logout) button to the bottom of the sidebar.
5. **Mobile Friendly Sidebar:** Prior to the login feature, we also removed the CSS lock on the sidebar and styled the native toggle button cleanly, so it now collapses automatically on mobile and can be toggled.

## What to do next (When you return)
1. **Test Locally:** Refresh your localhost app (`http://localhost:8501`) and verify that the login page appears. Try logging in with `admin` / `supervisor` and test the "Keluar" button in the sidebar.
2. **Commit & Push:** Once you verify the login works locally, we need to commit these changes in `app.py` and push them to GitHub.
3. **Streamlit Cloud Setup:** Go to your Streamlit Community Cloud dashboard for this app. Go to **Settings > Secrets** and paste the exact same credentials block so the live app can authenticate:
   ```toml
   [credentials]
   username = "admin"
   password = "supervisor"
   ```
4. **Mobile Verification:** You can also check the live app on your phone to see how the collapsible sidebar feels.

Just say "let's pick up from leftover.md" when you're back!
