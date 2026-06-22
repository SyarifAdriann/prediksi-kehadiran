# Training Pipeline Master

This document maps the exact technical methodology used to train the machine learning model in `02_train.py`. It serves as the master reference for how data flows from processed inputs into the final predictive artifact.

## 1. Input Data Processing
*   **Source**: `data/processed/dataset_gabungan.csv`
*   **Total Data Processed**: 300 rows.
*   **Class Imbalance Resolution**: At this stage, the data is relatively balanced at 60.0% Hadir (180 rows) vs 40.0% Tidak Hadir (120 rows).

## 2. Train-Test Split Strategy
The dataset is split to isolate a portion of the data exclusively for blind evaluation, ensuring the model's accuracy reflects its performance on unseen data.
*   **Method**: `train_test_split` (from Scikit-learn).
*   **Ratio**: 80% Training Data, 20% Testing Data.
*   **Stratification**: `stratify=y` ensures the 60/40 class distribution is perfectly maintained in both subsets.
*   **Resulting Train Set**: 240 rows (144 Hadir, 96 Tidak Hadir).
*   **Resulting Test Set**: 60 rows (36 Hadir, 24 Tidak Hadir).

## 3. GridSearchCV + StratifiedKFold — Pencarian Parameter Optimal
Instead of relying on arbitrary default settings, the pipeline employs a rigorous grid search to mathematically locate the most optimal model configuration.
*   **Parameter Grid (108 combinations tested)**:
    *   `n_estimators`: [100, 200, 300]
    *   `max_depth`: [None, 10, 20, 30]
    *   `min_samples_split`: [2, 5, 10]
    *   `min_samples_leaf`: [1, 2, 4]
*   **Scoring Metric**: `scoring="f1_macro"`
    *   *Why?* Optimizing for `f1_macro` forces the grid search to prioritize finding parameters that predict BOTH classes equally well, instead of just exploiting the 60% majority class to artificially inflate accuracy.
*   **Cross Validation (CV)**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
    *   *Why?* Each of the 108 combinations is evaluated 5 separate times on 5 different slices of the training data. This prevents the model from choosing a configuration that was simply "lucky" on a specific subset.
*   **Best Parameters Discovered**: 
    *   `n_estimators=300`
    *   `max_depth=None`
    *   `min_samples_split=5`
    *   `min_samples_leaf=2`

## 4. Final Model Training
The pipeline takes the best configuration discovered by GridSearchCV and applies it to the `RandomForestClassifier`.
*   **Algorithmic Weighting**: `class_weight="balanced"` is applied to heavily penalize errors made when predicting the minority "Tidak Hadir" class.
*   **Execution**: The algorithm grows 300 independent decision trees based on random subsets of features and data, combining their outputs via majority voting to combat overfitting.

## 5. Model Evaluation (Test Set Results)
The fully optimized model is finally pitted against the blind 60-row Test Set.
*   **Accuracy**: 83.33%
*   **Precision (Weighted)**: 83.25%
*   **Recall (Weighted)**: 83.33%
*   **F1-Score (Weighted)**: 83.20%
*   **Minority Class (Tidak Hadir) Success**: The model achieved an F1-Score of 78.26% on the minority class, proving the success of the GridSearchCV `f1_macro` optimization pipeline.

## 6. Output Artifacts Generation
1.  **`models/random_forest_model.pkl`**: The compiled, fully trained algorithm ready for deployment in the Streamlit app.
2.  **`models/best_params.json`**: A record of the exact winning parameters discovered by GridSearchCV.
3.  **`models/split_info.json`**: Mathematical records of the train/test splits.
4.  **`outputs/metrics_summary.json`**: The final scoring metrics, confusion matrix, and feature importances for use in application dashboards.
