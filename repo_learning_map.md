# Repository Reconnaissance Report: Attendance Prediction System

This document provides a comprehensive map of the thesis repository for educational purposes.

## 1. Repository Structure

### Folders
*   `data/`: Contains dataset files.
    *   `data/raw/`: Raw Excel files from the source. (Important)
    *   `data/processed/`: Cleaned and merged dataset ready for modeling. (Important)
*   `scripts/`: Contains the core machine learning pipeline scripts. (Critical)
*   `models/`: Stores the trained model and split metadata. (Critical)
*   `outputs/`: Stores evaluation metrics, charts, and visualizations. (Supporting)
*   `figures/`: Contains diagrams and illustrations for the thesis. (Supporting)
*   `.streamlit/`: Configuration and secrets for Streamlit deployment. (Supporting)

### Python Files
*   `scripts/01_preprocess.py`: Extracts raw data, cleans, standardizes, engineers features, and encodes them. (Critical)
*   `scripts/02_train.py`: Loads processed data, splits it, and trains the Random Forest model. (Critical)
*   `scripts/03_evaluate.py`: Generates evaluation metrics (accuracy, precision, recall, F1) and visualizes results. (Critical)
*   `app.py`: The Streamlit web application that serves the model and dashboard. (Critical)

### Data Files (CSV/XLSX)
*   `data/raw/event1_raw.xlsx`: Raw attendance data for Event 1. (Important)
*   `data/raw/event2_raw.xlsx`: Raw attendance data for Event 2. (Important)
*   `data/processed/dataset_gabungan.csv`: Merged and preprocessed dataset used for training. (Critical)

### Model & Output Files
*   `models/random_forest_model.pkl`: The serialized Random Forest classifier. (Critical)
*   `models/split_info.json`: Metadata regarding the train-test split sizes and class distributions. (Supporting)
*   `outputs/metrics_summary.json`: Detailed evaluation metrics stored in JSON format. (Supporting)
*   `outputs/*.png`: Evaluation charts (confusion matrix, feature importance, distributions). (Supporting)

### Configuration Files
*   `requirements.txt` & `environment.yml`: Python package dependency lists. (Important)
*   `.streamlit/config.toml`: Streamlit theme and server configuration. (Supporting)
*   `.streamlit/secrets.toml`: Local credentials for the admin login gate. (Important)

---

## 2. Actual ML Pipeline Discovery

The actual implemented machine learning pipeline proceeds linearly through three standalone scripts and one deployment application:

1.  **Raw Data Source:** Data starts in `event1_raw.xlsx` and `event2_raw.xlsx`.
2.  **Preprocessing:** 
    *   File: `scripts/01_preprocess.py`
    *   Function: `load_event()` uses `pandas.read_excel()` to load the data.
    *   Standardization functions like `standardize_kecamatan()` and `standardize_kab()` clean text columns.
3.  **Feature Engineering:**
    *   File: `scripts/01_preprocess.py`
    *   Function: `correct_jarak_outliers()` corrects anomalous distances using `haversine_km()`.
    *   Function: `engineer_hadir_event_sebelumnya()` creates a loyalty feature by matching names across events.
    *   Function: `encode_features()` maps string categories to integers.
4.  **Train/Test Split:**
    *   File: `scripts/02_train.py`
    *   Function: `train_test_split` from `sklearn.model_selection`.
    *   Variables: `X_train`, `X_test`, `y_train`, `y_test`.
5.  **Model Training:**
    *   File: `scripts/02_train.py`
    *   Class: `RandomForestClassifier` from `sklearn.ensemble`.
    *   Function: `model.fit(X_train, y_train)` trains the model.
6.  **Evaluation:**
    *   File: `scripts/03_evaluate.py`
    *   Functions: `accuracy_score()`, `precision_score()`, `recall_score()`, `f1_score()`, `confusion_matrix()` generate metrics.
    *   Variable: `model.feature_importances_` extracts feature importance.
7.  **Prediction & Streamlit Deployment:**
    *   File: `app.py`
    *   Variables: User inputs mapped to `X_input` dataframe.
    *   Functions: `model.predict(X_input)` and `model.predict_proba(X_input)` generate live predictions for the user interface.

---

## 3. Dataset Inventory

*   **Datasets Used:** 
    *   `event1_raw.xlsx` (Event 1)
    *   `event2_raw.xlsx` (Event 2)
    *   `dataset_gabungan.csv` (Merged and preprocessed)
*   **Target Variable:** `Status_Kehadiran` (1 = Hadir, 0 = Tidak Hadir)
*   **Columns Used For Modeling:** `Jenis_Kelamin`, `Usia`, `Jarak_km`, `Status_Pendaftaran`, `Event_ID`, `hadir_event_sebelumnya`.
*   **Highlight:** `hadir_event_sebelumnya`
    *   **Where it's created:** In `scripts/01_preprocess.py`, within the `engineer_hadir_event_sebelumnya(df)` function.
    *   **How it works:** It creates a lookup dictionary of `ID ANGGOTA` (names) from Event 1. For rows belonging to Event 2, it checks if the participant's name exists in the Event 1 lookup and if their status was "Hadir". If yes, the value is 1, otherwise 0. Event 1 rows always receive 0.

---

## 4. Feature Engineering Inventory

### 1. Distance Parsing and Outlier Correction (`Jarak_km`)
*   **Source Columns:** `Jarak`, `Kecamatan`
*   **Resulting Column:** `Jarak_km`
*   **Implementation:** `scripts/01_preprocess.py` in `parse_jarak()` and `correct_jarak_outliers()`.
*   **Exact Logic:** 
    *   `parse_jarak` uses Regex to extract numbers from strings like "12 KM" or "400 M".
    *   `correct_jarak_outliers` calculates the Haversine distance (`haversine_km`) from the participant's `Kecamatan` centroid to the event coordinates (`HALIM_LAT`, `HALIM_LON`). If the recorded distance is > 3x the calculated centroid distance AND the difference is > 10km, it flags it as an outlier and replaces it with the centroid distance plus random noise `np.random.uniform(-0.5, 1.5)`. Missing values are filled with the column median.

### 2. Historical Attendance (`hadir_event_sebelumnya`)
*   **Source Columns:** `Event_ID`, `ID ANGGOTA`, `Status Kehadiran`
*   **Resulting Column:** `hadir_event_sebelumnya`
*   **Implementation:** `scripts/01_preprocess.py` in `engineer_hadir_event_sebelumnya()`.
*   **Exact Logic:** Uses case-insensitive string matching on `ID ANGGOTA` to find Event 2 attendees who were present at Event 1.

### 3. Categorical Encoding
*   **Source Columns:** `Jenis Kelamin`, `Status Pendaftaran`, `Status Kehadiran`
*   **Resulting Columns:** `Jenis_Kelamin`, `Status_Pendaftaran`, `Status_Kehadiran`
*   **Implementation:** `scripts/01_preprocess.py` in `encode_features()`.
*   **Exact Logic:** Dictionary mapping. (e.g., "Perempuan" -> 0, "Laki-Laki" -> 1).

---

## 5. Random Forest Inventory

*   **Implementation File:** `scripts/02_train.py`
*   **Algorithm:** `RandomForestClassifier` (scikit-learn)
*   **Hyperparameters Used:**
    *   `n_estimators`: `100`
    *   `class_weight`: `"balanced"`
    *   `random_state`: `42` (`RANDOM_SEED`)
*   **Train/Test Split Settings:**
    *   `test_size`: `0.20`
    *   `random_state`: `42`
    *   `stratify`: `y` (Target variable `Status_Kehadiran`)

---

## 6. Evaluation Inventory

Located entirely in `scripts/03_evaluate.py`:

*   **Accuracy:** `accuracy_score(y_test, y_pred)` (Line 126)
*   **Precision:** `precision_score(y_test, y_pred, average="weighted", zero_division=0)` (Line 127) and per-class precision using `pos_label` (Lines 132, 135).
*   **Recall:** `recall_score(y_test, y_pred, average="weighted", zero_division=0)` (Line 128) and per-class recall (Lines 133, 136).
*   **F1-Score:** `f1_score(y_test, y_pred, average="weighted", zero_division=0)` (Line 129) and per-class F1-score (Lines 134, 137).
*   **Confusion Matrix:** `confusion_matrix(y_test, y_pred)` (Line 139). Visualized using Seaborn heatmap (Line 221).
*   **Feature Importance:** `model.feature_importances_` (Line 155). Visualized as a horizontal bar chart (Line 252).

---

## 7. Streamlit Inventory

*   **Entry Point:** `app.py`
*   **Security Gate:** A custom login form using `_check_login()` compares input against `st.secrets["credentials"]`. Uses `st.stop()` to block rendering.
*   **Tabs:** `tab_pred` (Prediksi Kehadiran), `tab_dash` (Dashboard), `tab_eval` (Evaluasi Model), `tab_log` (Log Prediksi).
*   **Prediction Flow Trace:**
    1.  User enters data in the left column using `st.selectbox` and `st.number_input`.
    2.  User clicks the `predict_btn` ("Prediksi Sekarang").
    3.  Inputs are manually encoded to integers (e.g., `jk_enc`, `event_enc`).
    4.  A single-row pandas DataFrame `X_input` is created.
    5.  `model.predict(X_input)` generates the class prediction.
    6.  `model.predict_proba(X_input)` generates the probability distributions.
    7.  Results are saved to `st.session_state['pred_data']` and appended to `st.session_state["pred_log"]`.
    8.  `st.rerun()` triggers a UI refresh, rendering the result cards and probability progress bars on the right column.

---

## 8. Thesis Consistency Check

Based on code analysis and artifacts (like `editgambar.md`), here are the discrepancies between the original thesis design and the actual code reality:

| Thesis Claim (Diagrams/Docs) | Code Reality (`app.py`, `scripts/`) | Match? |
| :--- | :--- | :--- |
| System saves predictions to a "Database" | Saves predictions temporarily to `st.session_state` (memory) and a local `pred_log.json` file. No relational database exists. | ❌ No |
| System "Cetak Laporan" (Prints Report) | Uses `st.download_button` to generate CSV or PDF downloads. No physical printing functionality. | ❌ No |
| Architecture uses XGBoost / SMOTE | Repository explicitly relies only on `RandomForestClassifier`. XGBoost scripts have been deleted. | ❌ No |
| Feature `hadir_event_sebelumnya` exists natively | Feature is manually engineered in Python via name-matching across two separate event files. | ❌ No |

---

## 9. Learning Difficulty Ranking

From most difficult to least difficult:

1.  **Outlier Correction & Geolocation Math (`correct_jarak_outliers`):** Requires understanding the Haversine formula, centroid coordinate mapping, boundary logic (3x distance + 10km difference), and random noise injection (`np.random.uniform`) to synthesize realistic data.
2.  **Cross-Event Feature Engineering (`engineer_hadir_event_sebelumnya`):** Requires understanding how to build lookup dictionaries, iterate through rows, and match string data (`ID ANGGOTA`) while handling edge cases (Event 1 vs Event 2).
3.  **Streamlit Session State & Logic Gates:** Understanding how Streamlit reruns the script top-to-bottom on every interaction, and why `st.session_state` is necessary to persist login status (`logged_in`) and prediction logs (`pred_log`) across reruns.
4.  **Machine Learning Evaluation Metrics:** Understanding the difference between weighted average and per-class metrics (`pos_label=1` vs `pos_label=0`), and interpreting the confusion matrix.
5.  **Random Forest Training (`02_train.py`):** The easiest part. It relies on standard `scikit-learn` boilerplate (`train_test_split`, `fit`, `dump`). The hyperparameters (`n_estimators`, `class_weight`) are simple and self-explanatory.
