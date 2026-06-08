"""
02_train.py
-----------
Tahap KDD: Pemodelan — Training Random Forest + GridSearchCV.

Input  : data/processed/dataset_gabungan.csv
Output : models/random_forest_model.pkl
         models/split_info.json
         models/best_params.json
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

RANDOM_SEED = 42
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(ROOT, "data", "processed")
MODEL_DIR = os.path.join(ROOT, "models")

INPUT_CSV  = os.path.join(DATA_PROC, "dataset_gabungan.csv")
OUTPUT_PKL = os.path.join(MODEL_DIR, "random_forest_model.pkl")
SPLIT_INFO = os.path.join(MODEL_DIR, "split_info.json")
BEST_PARAMS_JSON = os.path.join(MODEL_DIR, "best_params.json")

FEATURE_COLS = [
    "Jenis_Kelamin",
    "Usia",
    "Jarak_km",
    "Status_Pendaftaran",
    "Event_ID",
    "hadir_event_sebelumnya",
]
TARGET_COL = "Status_Kehadiran"
TEST_SIZE  = 0.20


def main():
    print("=" * 60)
    print("02_train.py — Random Forest + GridSearchCV")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    X  = df[FEATURE_COLS].copy()
    y  = df[TARGET_COL].copy()

    print(f"\n  Dataset: {len(df)} rows")
    print(f"  Hadir (1): {(y==1).sum()} | Tidak Hadir (0): {(y==0).sum()}")
    print(f"  Hadir %: {(y==1).mean()*100:.1f}%")

    # ── Train / Test Split ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )
    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Train Hadir: {(y_train==1).sum()} | Tidak Hadir: {(y_train==0).sum()}")
    print(f"  Test  Hadir: {(y_test==1).sum()}  | Tidak Hadir: {(y_test==0).sum()}")

    # ── GridSearchCV ───────────────────────────────────────────
    print("\n[STEP 1] GridSearchCV — mencari hyperparameter terbaik...")
    print("  (Ini bisa memakan 1-3 menit, harap tunggu...)")

    param_grid = {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
    }

    base_rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        scoring="f1_macro",       # optimize for balanced class performance
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    t0 = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - t0

    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    print(f"\n  GridSearchCV done in {elapsed:.1f}s")
    print(f"  Best CV F1-macro : {best_cv_score:.4f}")
    print(f"  Best params      : {best_params}")

    # Save best params
    best_params_out = {
        "best_params": best_params,
        "best_cv_f1_macro": round(float(best_cv_score), 4),
        "grid_search_elapsed_s": round(elapsed, 1),
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "scoring": "f1_macro",
        "cv_folds": 5,
    }
    with open(BEST_PARAMS_JSON, "w") as f:
        json.dump(best_params_out, f, indent=2)
    print(f"  Saved: {BEST_PARAMS_JSON}")

    # ── Final Model (best estimator already refitted on full train) ──
    print("\n[STEP 2] Model final dengan parameter terbaik...")
    model = grid_search.best_estimator_

    # Quick sanity check on test set
    y_pred_test = model.predict(X_test)
    test_acc    = accuracy_score(y_test, y_pred_test)
    test_f1     = f1_score(y_test, y_pred_test, average="macro")
    test_f1_tidak = f1_score(y_test, y_pred_test, pos_label=0, average="binary")

    print(f"  Test Accuracy   : {test_acc*100:.2f}%")
    print(f"  Test F1-macro   : {test_f1*100:.2f}%")
    print(f"  Test F1 Tidak Hadir: {test_f1_tidak*100:.2f}%")

    # ── Save model & split info ────────────────────────────────
    joblib.dump(model, OUTPUT_PKL)
    print(f"\n  Saved model: {OUTPUT_PKL}")

    split_info = {
        "test_size":         TEST_SIZE,
        "random_seed":       RANDOM_SEED,
        "feature_cols":      FEATURE_COLS,
        "target_col":        TARGET_COL,
        "n_train":           int(len(X_train)),
        "n_test":            int(len(X_test)),
        "train_hadir":       int((y_train == 1).sum()),
        "train_tidak_hadir": int((y_train == 0).sum()),
        "test_hadir":        int((y_test  == 1).sum()),
        "test_tidak_hadir":  int((y_test  == 0).sum()),
        "best_params":       best_params,
    }
    with open(SPLIT_INFO, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"  Saved split_info: {SPLIT_INFO}")

    print("\n[OK] Training selesai.\n")


if __name__ == "__main__":
    main()
