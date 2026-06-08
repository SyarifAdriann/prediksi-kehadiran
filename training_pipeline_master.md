# Master Training Pipeline Trace

This document provides an exhaustive, end-to-end trace of the machine learning pipeline implemented in this repository, mapping a single record from raw Excel input to final model prediction, alongside an analysis of methodological decisions.

---

## 1. Raw Source Files

The pipeline begins with two Excel files:
*   **File 1:** `data/raw/event1_raw.xlsx`
*   **File 2:** `data/raw/event2_raw.xlsx`
*   **Total Row Count:** 300 rows combined.
*   **Raw Columns (12):** `"No.", "ID ANGGOTA", "Usia", "Jenis Kelamin", "Alamat tempat tinggal", "Desa/Kelurahan", "Kecamatan", "Kabupaten/Kota", "Provinsi", "Jarak", "Status Pendaftaran", "Status Kehadiran"`

---

## 2. Preprocessing Walkthrough (Line-by-Line)

Executed in `scripts/01_preprocess.py`. The `No.` column is dropped immediately.

### Transformation 1: String Standardization
*   **Code Location:** `standardize_kecamatan(val)`, `standardize_kab(val)`
*   **Original Value:** `" Jakarta Utata "`
*   **Transformed Value:** `"Jakarta Utara"`
*   **Reason:** Fixes typos, strips trailing/leading whitespaces, and standardizes casing so that regional groupings are consistent.

### Transformation 2: Missing Value Handling (Categorical)
*   **Code Location:** `df[col] = df[col].replace("nan", None)`
*   **Original Value:** `"nan"` (string representation of pandas NaN)
*   **Transformed Value:** `None`
*   **Reason:** Ensures missing categorical values are treated as actual nulls rather than the literal string "nan".

### Transformation 3: Distance Parsing (`parse_jarak`)
*   **Code Location:** `parse_jarak(val)`
*   **Original Value:** `"13,8 KM"` or `"400 M"`
*   **Transformed Value:** `13.8` or `0.4`
*   **Reason:** Converts inconsistent human-entered text strings containing units and comma-decimals into standard float values in kilometers.

---

## 3. Deep Dive: `hadir_event_sebelumnya`

This is a custom engineered feature representing participant loyalty.

*   **Implementation Location:** `engineer_hadir_event_sebelumnya(df)` in `01_preprocess.py`
*   **Lookup Logic:**
    1. It isolates all rows belonging to `Event 1`.
    2. It creates a dictionary mapping the lowercased, stripped `ID ANGGOTA` to their `Status Kehadiran`.
*   **Matching Logic:**
    *   For any row where `Event_ID == 1`, the value is forced to `0` (since there is no preceding event).
    *   For any row where `Event_ID == 2`, it takes the `ID ANGGOTA`, looks it up in the Event 1 dictionary.
    *   If found AND status was `"Hadir"`, value becomes `1`. Otherwise, `0`.
*   **Example Record Trace:**
    *   *Raw:* Budi attends Event 1 (Status: Hadir).
    *   *Raw:* Budi registers for Event 2.
    *   *Transformation:* When processing Budi's Event 2 row, the code finds "budi" in the Event 1 dictionary with status "Hadir". Budi's Event 2 row receives `hadir_event_sebelumnya = 1`.

---

## 4. Deep Dive: `Jarak_km` (Haversine & Outliers)

Because participant-entered distances were highly unreliable, a spatial correction algorithm was implemented.

*   **Implementation Location:** `correct_jarak_outliers(df)` in `01_preprocess.py`
*   **Haversine Calculation:** It uses `haversine_km(lat1, lon1, lat2, lon2)` to calculate the exact spherical distance between the event coordinates (Halim) and the hardcoded centroid coordinates of the participant's `Kecamatan`.
*   **Outlier Detection Rules:** 
    An inputted distance is flagged as an outlier IF:
    1. Input Distance > `3 * Centroid Distance` AND
    2. Input Distance > `Centroid Distance + 10 km`
*   **Replacement & Noise Generation:**
    *   If flagged, the input distance is discarded.
    *   It is replaced by: `Centroid Distance + np.random.uniform(-0.5, 1.5)`
    *   *Reasoning:* Replacing every outlier in a specific Kecamatan with the exact same centroid distance would create massive artificial spikes in the data distribution. The `uniform(-0.5, 1.5)` noise introduces slight variance to mimic realistic household distributions around a district center. Missing values are filled with the dataset median.

---

## 5. Train/Test Split Details

Executed in `scripts/02_train.py` using `train_test_split(..., stratify=y)`.

*   **Total Dataset:** 300 rows
*   **Split Ratio:** 80% Train / 20% Test
*   **Training Set:** 240 rows
    *   Hadir: 189
    *   Tidak Hadir: 51
*   **Test Set:** 60 rows
    *   Hadir: 47
    *   Tidak Hadir: 13
*   *Note:* The `stratify=y` argument ensures the 78.6% vs 21.4% class imbalance is perfectly preserved across both splits.

---

## 6. Single Record Trace (End-to-End)

Let's follow a hypothetical participant through the pipeline:

1.  **Raw Row:**
    *   `Jenis Kelamin`: `"Perempuan"`
    *   `Usia`: `25`
    *   `Kecamatan`: `"Makassar"`
    *   `Jarak`: `"50 KM"` (Participant exaggerated)
    *   `Event_ID`: `2` (Attended Event 1 previously)
2.  **Preprocessing:** 
    *   `Kecamatan` standardized to `"Makasar"`. 
    *   `Jarak` parsed to `50.0`.
3.  **Feature Engineering:**
    *   Algorithm sees 50km for Makasar. Centroid distance for Makasar to Halim is actually ~2km.
    *   50 > (3 * 2) AND 50 > (2 + 10). Outlier triggered!
    *   Distance replaced with `2.0 + noise (e.g., 0.8)` = `2.8 km`.
    *   Lookup confirms attendance at Event 1. `hadir_event_sebelumnya` = `1`.
4.  **Encoded Values:**
    *   `Jenis_Kelamin` mapped `"Perempuan"` → `0`.
5.  **Model Input Vector:**
    *   `X_input` = `[0, 25, 2.8, 1, 2, 1]` *(Gender, Age, Dist, RegStatus, Event, Loyalty)*
6.  **Prediction:**
    *   Vector passes through Random Forest trees.
    *   `predict_proba()` returns `[0.12, 0.88]`.
    *   `predict()` returns `1` (Hadir).

---

## 7. Potential Examiner Challenges

If auditing this repository, examiners could raise the following methodological questions:

### 1. Dataset Size is Too Small
*   **Criticism:** 300 rows across 2 events is very small for Random Forest, risking high variance. The test set is only 60 rows; a change of just 3 predictions shifts accuracy by 5%.
*   **Defense:** Valid criticism. For a thesis, 300 rows is acceptable if the data was collected manually (primary data). The use of `class_weight="balanced"` and `stratify=y` mitigates some small-sample bias.

### 2. Random Noise in Distance Imputation
*   **Criticism:** Injecting `np.random.uniform(-0.5, 1.5)` into outlier distances alters deterministic spatial data.
*   **Defense:** Partially valid. While statistically unorthodox, it prevents the Random Forest from creating an artificially dense node at exact centroid values, smoothing the decision boundaries.

### 3. Leakage in Event 1 `hadir_event_sebelumnya`
*   **Criticism:** All participants in Event 1 are hardcoded to `hadir_event_sebelumnya = 0` because there is no prior event. This means `0` means two different things: "First event" (for Event 1) AND "Did not attend" (for Event 2).
*   **Defense:** Valid. A better approach would be to use Event 1 *only* as historical lookup data and train the model exclusively on Event 2. However, with only 300 rows total, discarding Event 1 from training would halve the dataset. The inclusion of `Event_ID` as a feature allows the tree to split logic based on which event is occurring, mitigating the semantic collision.

### 4. Overfitting via Hyperparameter Defaulting
*   **Criticism:** The code uses hardcoded `n_estimators=100` without performing GridSearchCV or randomized hyperparameter tuning.
*   **Defense:** Valid. Earlier versions of the repository (now deleted) contained an `04_experiment.py` script that did run `GridSearchCV`. The author likely hardcoded the optimal parameters discovered during that experiment into the final `02_train.py` script for simplicity of deployment.
