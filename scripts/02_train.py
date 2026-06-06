"""
02_train.py
-----------
Tahap KDD: Pemodelan — Training Random Forest Classifier.

Input  : data/processed/dataset_gabungan.csv
Output : models/random_forest_model.pkl

Jalankan dari root folder skripsi/:
    python scripts/02_train.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ============================================================
# KONSTANTA
# ============================================================
RANDOM_SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(ROOT, "data", "processed")
MODEL_DIR = os.path.join(ROOT, "models")

INPUT_CSV = os.path.join(DATA_PROC, "dataset_gabungan.csv")
OUTPUT_PKL = os.path.join(MODEL_DIR, "random_forest_model.pkl")
SPLIT_INFO = os.path.join(MODEL_DIR, "split_info.json")

FEATURE_COLS = [
    "Jenis_Kelamin",
    "Usia",
    "Jarak_km",
    "Status_Pendaftaran",
    "Event_ID",
    "hadir_event_sebelumnya",
]
TARGET_COL = "Status_Kehadiran"

TEST_SIZE = 0.20


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("02_train.py — Training Random Forest Classifier")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ----------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------
    print("\n[STEP 1] Load Dataset")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Shape: {df.shape}")

    # Validasi kolom yang dibutuhkan
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    print(f"  Fitur  : {FEATURE_COLS}")
    print(f"  Target : {TARGET_COL}")
    print(f"  Distribusi target:")
    vc = y.value_counts()
    for val, cnt in vc.items():
        label = "Hadir" if val == 1 else "Tidak Hadir"
        print(f"    {label} ({val}): {cnt} ({cnt/len(y)*100:.1f}%)")

    # ----------------------------------------------------------
    # Train-Test Split
    # ----------------------------------------------------------
    print(f"\n[STEP 2] Train-Test Split (80/20, stratified, seed={RANDOM_SEED})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )
    print(f"  Training set : {len(X_train)} baris")
    print(f"  Test set     : {len(X_test)} baris")
    print(f"  Train — Hadir: {(y_train==1).sum()} | Tidak Hadir: {(y_train==0).sum()}")
    print(f"  Test  — Hadir: {(y_test==1).sum()}  | Tidak Hadir: {(y_test==0).sum()}")

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------
    print(f"\n[STEP 3] Training RandomForestClassifier")
    print(f"  Parameters: n_estimators=100, random_state={RANDOM_SEED}, class_weight='balanced'")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    print(f"  Training selesai. n_classes = {model.n_classes_}")
    print(f"  Classes: {model.classes_}")

    # ----------------------------------------------------------
    # Simpan model
    # ----------------------------------------------------------
    print(f"\n[STEP 4] Simpan Model")
    joblib.dump(model, OUTPUT_PKL)
    print(f"  Saved: {OUTPUT_PKL}")

    # Simpan split info (untuk dipakai oleh 03_evaluate.py)
    split_info = {
        "test_size": TEST_SIZE,
        "random_seed": RANDOM_SEED,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_hadir": int((y_train == 1).sum()),
        "train_tidak_hadir": int((y_train == 0).sum()),
        "test_hadir": int((y_test == 1).sum()),
        "test_tidak_hadir": int((y_test == 0).sum()),
    }
    with open(SPLIT_INFO, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"  Saved: {SPLIT_INFO}")

    # ----------------------------------------------------------
    # Quick accuracy check
    # ----------------------------------------------------------
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"\n[STEP 5] Quick Accuracy Check")
    print(f"  Training accuracy : {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Test accuracy     : {test_acc:.4f} ({test_acc*100:.2f}%)")

    print("\n[OK] Training selesai. Jalankan 03_evaluate.py untuk evaluasi lengkap.\n")


if __name__ == "__main__":
    main()
