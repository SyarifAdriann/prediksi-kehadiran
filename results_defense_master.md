# Results & Defense Master Document

## 1. Verified Model Metrics (v2 - GridSearchCV)

### Dataset Overview
*   **Total Data**: 300
*   **Hadir**: 180 (60.00%) — *Majority Baseline*
*   **Tidak Hadir**: 120 (40.00%)
*   **Train Set**: 240 (Hadir: 144, Tidak Hadir: 96)
*   **Test Set**: 60 (Hadir: 36, Tidak Hadir: 24)

### Performance Metrics (Test Set = 60)
*   **Accuracy**: 83.33%
*   **Precision (Weighted)**: 83.25%
*   **Recall (Weighted)**: 83.33%
*   **F1-Score (Weighted)**: 83.20%

#### Per-Class Breakdown
*   **Kelas "Hadir"** (Support: 36):
    *   Precision: 84.21%
    *   Recall: 88.89%
    *   F1-Score: 86.49%
*   **Kelas "Tidak Hadir"** (Support: 24):
    *   Precision: 81.82%
    *   Recall: 75.00%
    *   F1-Score: 78.26%

#### Confusion Matrix
*   **True Positive (TP)** = 32 (Diprediksi Hadir, Aktual Hadir)
*   **True Negative (TN)** = 18 (Diprediksi Tidak Hadir, Aktual Tidak Hadir)
*   **False Positive (FP)** = 6 (Diprediksi Hadir, Aktual Tidak Hadir)
*   **False Negative (FN)** = 4 (Diprediksi Tidak Hadir, Aktual Hadir)

### Feature Importance Ranking
1.  **Jarak_km**: 58.58%
2.  **Usia**: 26.12%
3.  **hadir_event_sebelumnya**: 11.60%
4.  **Event_ID**: 2.35%
5.  **Jenis_Kelamin**: 1.25%
6.  **Status_Pendaftaran**: 0.10%

### Model & Training Configuration
*   **Algorithm**: Random Forest Classifier
*   **Optimization**: GridSearchCV (108 parameter combinations tested)
*   **Cross Validation**: StratifiedKFold(n_splits=5)
*   **Scoring Metric**: `f1_macro` (Optimization focused on balancing both classes, not just raw accuracy)
*   **Best Parameters Found**: 
    *   `n_estimators`: 300
    *   `max_depth`: None (null)
    *   `min_samples_split`: 5
    *   `min_samples_leaf`: 2
    *   `class_weight`: "balanced"
    *   `random_state`: 42

---

## 2. Examiner Q&A (Defense Strategy)

The following arguments have been explicitly formulated to neutralize common attacks during the thesis defense.

**Q1: The Baseline Attack — "Is your model actually doing anything? If I just guess 'Hadir' every time, what accuracy do I get?"**
*   **Attack Neutralized**: The dataset is split 60% Hadir and 40% Tidak Hadir. Therefore, the "Majority Baseline" (guessing 'Hadir' blindly) only yields **60.00%** accuracy.
*   **Defense**: Our model achieves **83.33%** accuracy. This means our model **BEATS the baseline by a massive +23.33% margin**. It proves that the model is intelligently learning the patterns of the data (like Distance and Age), not just making blind guesses.

**Q2: The Class Imbalance / Minority Class Weakness — "Can your model actually detect the 'Tidak Hadir' class, or is it biased?"**
*   **Defense**: In previous iterations, detecting the minority class was a weakness (F1-score was 21%). However, in this final version, the F1-score for the "Tidak Hadir" class has improved drastically to **78.26%**, with a Recall of 75.00%.
*   This is now a **STRENGTH**. We successfully mitigated the minority class bias by utilizing `GridSearchCV` optimized on the `f1_macro` metric, ensuring the model prioritizes identifying absent members just as much as present members.

**Q3: Methodology Justification — "Why Random Forest? Did you even tune it?"**
*   **Defense**: We didn't just use a default algorithm out of the box. We implemented **GridSearchCV**, which systematically tested **108 different hyperparameter combinations** (varying trees from 100 to 300, depth limits, and leaf splits). 
*   To ensure the results were robust and not just lucky, we combined this search with **Stratified 5-Fold Cross Validation (`StratifiedKFold(n_splits=5)`)**. This rigorously validated each of the 108 combinations 5 times across different proportional data folds before selecting the absolute best configuration (`n_estimators=300`, `min_samples_split=5`, etc.).

**Q4: Data Balancing Strategy — "How did you handle the imbalanced data?"**
*   **Defense**: We utilized a multi-layered approach to balancing. First, at the dataset level, we performed data augmentation/balancing to adjust the extreme ratio to a much healthier **60/40** distribution. Second, at the algorithm level, we applied the `class_weight='balanced'` parameter. This mathematically penalizes the model heavier if it incorrectly predicts the minority class, forcing it to pay attention to the "Tidak Hadir" patterns. Finally, our GridSearch specifically optimized the `f1_macro` score instead of standard accuracy.

**Q5: "Why is `hadir_event_sebelumnya` so important now?"**
*   **Defense**: Feature importance analysis revealed it is the 3rd most powerful predictor (11.60%). This is highly logical: past behavior is a strong indicator of future behavior. A member who demonstrated the commitment to attend Event 1 has a statistically measurable higher probability of attending Event 2. This justifies our decision to analyze longitudinal data across 2 events instead of just looking at 1 isolated event.
