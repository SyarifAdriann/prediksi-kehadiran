# Thesis Defense Audit & Results Master

**Project Status:** High-Risk  
**Focus:** Repository Evidence, Quantitative Validation, Methodological Scrutiny

This document acts as an uncompromising defense audit for the Attendance Prediction System. It is designed to prepare the candidate for extreme scrutiny by a knowledgeable examiner.

---

## 1. Locate All Result Sources

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

## 2. Extract Actual Model Performance

Based entirely on `outputs/metrics_summary.json` (Test Set Size = 60):

*   **Final Thesis Model:** Random Forest Classifier (XGBoost and SMOTE iterations were deleted from the repository).
*   **Accuracy:** 75.0%
*   **Precision (Weighted):** 69.6%
*   **Recall (Weighted):** 75.0%
*   **F1-score (Weighted):** 71.26%

*Note: The metrics report "Weighted" averages, which mask poor performance on the minority class. We must look at the per-class metrics to see the reality.*

---

## 3. Confusion Matrix Deep Audit

Based on `metrics_summary.json`:

*   **True Positives (TP):** 43 (Actual Hadir, Predicted Hadir)
*   **True Negatives (TN):** 2 (Actual Tidak Hadir, Predicted Tidak Hadir)
*   **False Positives (FP):** 11 (Actual Tidak Hadir, Predicted Hadir)
*   **False Negatives (FN):** 4 (Actual Hadir, Predicted Tidak Hadir)

**Mistake Profile:**
The model heavily overpredicts "Hadir". It missed 11 out of 13 absent people, guessing they would attend. 

**Calculations:**
*   **Sensitivity / Recall (Hadir):** `43 / (43 + 4) = 91.49%` (Excellent at finding attendees)
*   **Specificity / Recall (Tidak Hadir):** `2 / (2 + 11) = 15.38%` (Terrible at finding absentees)
*   **False Positive Rate:** `11 / 13 = 84.62%`
*   **False Negative Rate:** `4 / 47 = 8.51%`

---

## 4. Majority-Class Baseline Test (Brutally Honest)

If an examiner asks: *"Why use Machine Learning? What if I just guessed everyone will attend?"*

*   Total Dataset: 300 rows
*   Hadir Count: 236 (78.67%)
*   Tidak Hadir Count: 64 (21.33%)

**Baseline "Guess Hadir" Accuracy: 78.67%**
**Random Forest Accuracy: 75.00%**

**Verdict:** The Random Forest model **loses to the naive baseline by 3.67%**. 
*Why?* Because the model tries to identify "Tidak Hadir" (and successfully finds 2 of them), but makes 11 false positive mistakes doing so, dropping its overall accuracy below the baseline. The improvement is NOT in accuracy, it is in moving Specificity from 0% (baseline) to 15% (Random Forest).

---

## 5. Feature Importance Audit

Ranked from `model.feature_importances_`:

| Rank | Feature | Importance | Business Logic |
| :--- | :--- | :--- | :--- |
| 1 | `Jarak_km` | 47.73% | Logical. Commute friction heavily dictates physical attendance. |
| 2 | `Usia` | 36.00% | Logical. Age cohorts have different availabilities/health. |
| 3 | `Status_Pendaftaran` | 6.51% | Weak. Registration intent should logically be #1 or #2. |
| 4 | `Event_ID` | 4.22% | Indicates slight variance between the Nov 09 and Nov 16 events. |
| 5 | `hadir_event_sebelumnya` | 3.35% | Surprisingly low. Loyalty/history is usually a top predictor. |
| 6 | `Jenis_Kelamin` | 2.18% | Logical. Gender rarely dictates attendance in general events. |

**Audit finding:** The model is almost entirely reliant on continuous spatial/demographic data (`Jarak_km` and `Usia` combine for 83.7%). Behavioral data (`Pendaftaran`, `Riwayat`) barely register.

---

## 6. Prediction Distribution Audit

*   Actual Test Set: 47 Hadir (78%), 13 Tidak Hadir (22%)
*   Model Predictions: 54 Hadir (90%), 6 Tidak Hadir (10%)

**Class Bias:** The model is highly biased toward the majority class. Despite using `class_weight="balanced"` in `scripts/02_train.py`, the random forest trees overwhelmingly vote for "Hadir". 

---

## 7. Misclassification Analysis

**Common Pattern for FP (Predicted Hadir, Actual Tidak Hadir):**
Because `Jarak_km` and `Usia` control 83% of the decision, if a participant lives close by (e.g., < 5km) and is in the average age bracket, the model is almost guaranteed to predict "Hadir". If that person suddenly gets sick or has an emergency, the model will output a False Positive because it lacks behavioral features to detect unexpected absences.

---

## 8. Leakage Investigation (CRITICAL)

**Methodology Audit:**
1. Event 1 (150 rows) and Event 2 (150 rows) are merged in `01_preprocess.py`.
2. The same people attend both events.
3. `train_test_split(test_size=0.2)` shuffles all 300 rows randomly.

**IID Violation (Independent and Identically Distributed):**
If Participant "Budi" is in Event 1 and Event 2, his Event 1 row might go to Train, and Event 2 row to Test. The model learns Budi's exact `Jarak_km` and `Usia` in Train, and predicts on the same `Jarak_km` and `Usia` in Test. The model is memorizing individuals, not generalizing.

**Time Travel Leakage:**
Because Event 1 (Past) and Event 2 (Future) are shuffled together:
The model is trained on Future data (Event 2) to predict Past data (Event 1 in the test set). Furthermore, Event 2 features contain `hadir_event_sebelumnya`, meaning the target label of Event 1 is literally encoded as a feature for Event 2 rows.

**Leakage Risk:** **HIGH**
*Defense required:* The student must admit that random splitting on panel data (repeated participants) is flawed, and that a time-based split (Train on Event 1, Test on Event 2) is the correct approach for future iterations.

---

## 9. Robustness Audit

*   **Sample Size:** 300 rows total. 60 rows for testing. *Weakness:* 60 rows is too small to prove statistical significance. A change of 1 prediction alters accuracy by 1.6%.
*   **Imbalance:** 78% vs 21%. *Weakness:* The model heavily overfits the majority class.
*   **Engineered Features:** `Jarak_km` adds random noise `np.random.uniform(-0.5, 1.5)` to centroid distances. *Weakness:* Injecting random noise into test set evaluation breaks determinism.

---

## 10. Examiner Attack Scenarios

### Q1: Why is your model's accuracy (75%) worse than if I just guessed "Hadir" for everyone (78%)?
*   *Why they ask:* This is the ultimate test of whether the student understands baselines.
*   *Strong Defense:* "Benar, Pak/Bu. Secara akurasi mentah, model kalah dari tebakan mayoritas. Tapi tebakan mayoritas memiliki Specificity 0%—artinya kita tidak bisa mendeteksi satupun orang yang Tidak Hadir. Model ini mengorbankan sedikit akurasi keseluruhan demi mendapatkan Specificity 15%, yaitu kemampuan mendeteksi sebagian orang yang akan absen, yang bernilai lebih tinggi secara operasional."
*   *Weak Answer:* "Karena datanya kurang." (Shows lack of metric understanding).

### Q2: Why did you use `train_test_split` acak (random) padahal datanya berurutan waktu (Event 1 lalu Event 2)?
*   *Why they ask:* Testing for time-travel leakage and IID violations.
*   *Strong Defense:* "Ini adalah batasan (limitation) dari penelitian ini. Karena total data hanya 300, memotong berdasarkan waktu (Event 1 untuk train, Event 2 untuk test) akan membuat data latih terlalu sedikit (150 baris). Jadi saya melakukan random split. Saya menyadari ini menimbulkan risiko IID violation karena peserta yang sama bisa ada di train dan test."

### Q3: Fitur `Jarak_km` mendominasi 47%. Tapi jaraknya dikoreksi menggunakan `random noise`. Bukankah ini memanipulasi data?
*   *Why they ask:* Questioning the preprocessing integrity.
*   *Strong Defense:* "Noise tersebut sangat kecil (-0.5 sampai 1.5 km) dan hanya ditambahkan pada centroid kecamatan karena data jarak dari peserta terbukti banyak yang tidak valid (outlier ekstrem). Tujuannya bukan memanipulasi, tapi mencegah model Random Forest membuat over-split pada satu titik nilai eksak centroid."

### Q4: Anda bilang pakai SMOTE dan XGBoost di proposal, mana di sistem akhir?
*   *Why they ask:* Checking alignment between thesis doc and codebase.
*   *Strong Defense:* "Setelah eksperimen di fase pengembangan, XGBoost dan SMOTE dibuang dari sistem final (`app.py`) karena Random Forest standar dengan `class_weight='balanced'` memberikan kompromi terbaik antara kecepatan komputasi di Streamlit dan hasil, tanpa menambah kompleksitas dependency."

### Q5: Nilai F1-Score kelas "Tidak Hadir" hanya 21%. Apakah model ini bisa diandalkan?
*   *Why they ask:* Checking if the student is blindly confident or realistically critical.
*   *Strong Defense:* "Belum cukup handal untuk produksi skala besar, Pak/Bu. Recall untuk Tidak Hadir hanya 15%. Ini terjadi karena ketidakseimbangan kelas (imbalance) yang parah dan kurangnya fitur prediktif perilaku (hanya mengandalkan jarak dan usia). Model ini adalah proof-of-concept yang butuh lebih banyak data absensi."

---

## 11. Final Verdict

*   **Technical Quality:** 8/10 (Excellent Streamlit UI, modular code, clean architecture).
*   **Methodological Quality:** 3/10 (High leakage risk, model loses to majority baseline, poor minority recall).
*   **Defense Readiness:** 4/10 (The student is walking into a trap if they focus on "75% Accuracy" without realizing it loses to the 78% baseline).

**Classification:** **HIGH RISK**

**Recommendation:** The student MUST NOT brag about accuracy during the presentation. The defense presentation must focus on **"Membangun Sistem (Engineering)"** rather than **"Kecerdasan Model (Data Science)"**. Acknowledge the ML flaws proactively as *Saran untuk penelitian selanjutnya*.
