# System Overview: PT Cahaya Ladara Nusantara Attendance Prediction

## 1. Project Objective
To develop a predictive system capable of forecasting member attendance (Hadir / Tidak Hadir) for the "Sushi & Onigiri" viral food training events organized by PT Cahaya Ladara Nusantara (PT CLN). The system uses machine learning to identify logistical and demographic patterns that influence a member's decision to attend.

## 2. Dataset Architecture
*   **Total Records**: 300 data points (representing 150 active members across 2 distinct training events).
*   **Class Distribution**: 60.0% Hadir (180 data), 40.0% Tidak Hadir (120 data).
*   **Target Variable**: `Status_Kehadiran` (1 = Hadir, 0 = Tidak Hadir).
*   **Predictor Features (6)**: 
    1.  `Jenis_Kelamin` (Binary)
    2.  `Usia` (Numeric)
    3.  `Jarak_km` (Numeric)
    4.  `Status_Pendaftaran` (Binary)
    5.  `Event_ID` (Categorical)
    6.  `hadir_event_sebelumnya` (Binary - Engineered Feature)

## 3. Data Processing Pipeline
1.  **Ingestion**: Reads raw `.xlsx` files using `pandas` and `openpyxl`.
2.  **Cleaning**: Standardizes district/city names (e.g., "Jakarta Utata" to "Jakarta Utara").
3.  **Distance Parsing & Correction**: Uses Regex to convert varied string formats ("12 KM", "400 M") into standard numeric floats (kilometers). Implements the **Haversine formula** to calculate geographic centroid coordinates as a fallback mechanism to correct illogical outlier distances.
4.  **Feature Engineering**: Cross-references participant names between Event 1 and Event 2 to dynamically generate the `hadir_event_sebelumnya` feature, tracking longitudinal attendance loyalty.
5.  **Label Encoding**: Converts all categorical text into machine-readable numeric binary formats.

## 4. Machine Learning Model (v2)
*   **Core Algorithm**: `RandomForestClassifier` (Scikit-learn).
*   **Optimization Engine**: Implements **GridSearchCV** combined with **StratifiedKFold(n_splits=5)** cross-validation to rigorously test 108 different hyperparameter combinations.
*   **Optimization Target**: Uses `scoring="f1_macro"` to force the model to balance its predictive capabilities equally between the majority and minority classes.
*   **Final Parameters**: `n_estimators=300`, `max_depth=None`, `min_samples_split=5`, `min_samples_leaf=2`, `class_weight="balanced"`.

## 5. System Performance & Metrics
*   **Accuracy**: **83.33%** (Beats the 60.00% majority baseline by +23.33%).
*   **Precision (Weighted)**: 83.25%
*   **Recall (Weighted)**: 83.33%
*   **F1-Score (Weighted)**: 83.20%
*   **Minority Class (Tidak Hadir) F1-Score**: 78.26% (A massive improvement demonstrating the success of the GridSearchCV balancing strategy).

## 6. Feature Importance Rankings
The model relies heavily on geographic logistics to make decisions:
1.  **Jarak_km**: 58.58% (Primary blocker)
2.  **Usia**: 26.12%
3.  **hadir_event_sebelumnya**: 11.60% (Proves longitudinal tracking was successful)
4.  **Event_ID**: 2.35%
5.  **Jenis_Kelamin**: 1.25%
6.  **Status_Pendaftaran**: 0.10%

## 7. Web Application Interface
*   **Framework**: Streamlit.
*   **Architecture**: Serverless/Database-less design. Relies entirely on static CSV/JSON files and dynamic `session_state` memory to run locally.
*   **Features**: Interactive data visualizer (Dashboard), manual prediction input form, dynamic evaluation metrics display, and a downloadable prediction log cache.
