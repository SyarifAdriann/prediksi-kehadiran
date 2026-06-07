"""
04_experiment.py
-----------------
Eksperimen peningkatan akurasi:
  Phase 1 — GridSearchCV (Random Forest tuning)
  Phase 2 — XGBoost (comparison)
  Phase 3 — SMOTE + best model (jika masih kurang)

Jalankan dari root folder skripsi/:
    python scripts/04_experiment.py

Output:
  - Tabel perbandingan semua model
  - Menyimpan model terbaik ke models/random_forest_model.pkl
  - Update models/split_info.json & outputs/metrics_summary.json
"""

import os
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)

warnings.filterwarnings("ignore")

# ============================================================
# KONSTANTA
# ============================================================
RANDOM_SEED = 42
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV   = os.path.join(ROOT, "data", "processed", "dataset_gabungan.csv")
MODEL_DIR   = os.path.join(ROOT, "models")
OUTPUT_PKL  = os.path.join(MODEL_DIR, "random_forest_model.pkl")
SPLIT_INFO  = os.path.join(MODEL_DIR, "split_info.json")
METRICS_JSON = os.path.join(ROOT, "outputs", "metrics_summary.json")

FEATURE_COLS = [
    "Jenis_Kelamin", "Usia", "Jarak_km",
    "Status_Pendaftaran", "Event_ID", "hadir_event_sebelumnya",
]
TARGET_COL = "Status_Kehadiran"

# ============================================================
# HELPERS
# ============================================================

def banner(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def evaluate(name, model, X_test, y_test):
    """Return dict of metrics for a fitted model."""
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1_w   = f1_score(y_test, y_pred, average="weighted")
    f1_0   = f1_score(y_test, y_pred, average=None)[0]   # Tidak Hadir
    f1_1   = f1_score(y_test, y_pred, average=None)[1]   # Hadir
    prec_w = precision_score(y_test, y_pred, average="weighted")
    rec_w  = recall_score(y_test, y_pred, average="weighted")
    cm     = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "name"     : name,
        "accuracy" : acc,
        "f1_weighted": f1_w,
        "f1_hadir" : f1_1,
        "f1_tidak" : f1_0,
        "precision": prec_w,
        "recall"   : rec_w,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }

def print_results(results):
    header = f"{'Model':<35} {'Accuracy':>9} {'F1-W':>7} {'F1-Hadir':>10} {'F1-TdkHdr':>11}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        marker = " ◀ BEST" if r.get("best") else ""
        print(
            f"{r['name']:<35} "
            f"{r['accuracy']*100:>8.2f}% "
            f"{r['f1_weighted']*100:>6.2f}% "
            f"{r['f1_hadir']*100:>9.2f}% "
            f"{r['f1_tidak']*100:>10.2f}%"
            f"{marker}"
        )

# ============================================================
# LOAD DATA
# ============================================================
banner("LOAD DATA")
df = pd.read_csv(INPUT_CSV)
X  = df[FEATURE_COLS].copy()
y  = df[TARGET_COL].copy()
print(f"  Shape: {df.shape}")
print(f"  Distribusi: Hadir={( y==1).sum()} ({(y==1).mean()*100:.1f}%)  "
      f"Tidak Hadir={(y==0).sum()} ({(y==0).mean()*100:.1f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)
print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

results = []

# ============================================================
# PHASE 0 — BASELINE (current RF, defaults)
# ============================================================
banner("PHASE 0 — Baseline Random Forest (defaults)")
t0 = time.time()
rf_base = RandomForestClassifier(
    n_estimators=100, random_state=RANDOM_SEED, class_weight="balanced"
)
rf_base.fit(X_train, y_train)
r = evaluate("RF Baseline (n=100, default)", rf_base, X_test, y_test)
r["time"] = time.time() - t0
results.append(r)
print(f"  Accuracy: {r['accuracy']*100:.2f}%  |  F1-Weighted: {r['f1_weighted']*100:.2f}%")
print(f"  F1 Hadir: {r['f1_hadir']*100:.2f}%  |  F1 Tidak Hadir: {r['f1_tidak']*100:.2f}%")
print(f"  Time: {r['time']:.1f}s")

# ============================================================
# PHASE 1 — GridSearchCV on Random Forest
# ============================================================
banner("PHASE 1 — GridSearchCV: Random Forest")

param_grid = {
    "n_estimators"     : [100, 200, 300, 500],
    "max_depth"        : [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf" : [1, 2, 4],
    "max_features"     : ["sqrt", "log2"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

n_combos = (len(param_grid["n_estimators"]) * len(param_grid["max_depth"]) *
            len(param_grid["min_samples_split"]) * len(param_grid["min_samples_leaf"]) *
            len(param_grid["max_features"]))
print(f"  Grid combinations: {n_combos}")

print("  Running GridSearch (this may take a few minutes)...")

t0 = time.time()
gs_rf = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced"),
    param_grid,
    cv=cv,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=0,
)
gs_rf.fit(X_train, y_train)
elapsed = time.time() - t0

best_rf   = gs_rf.best_estimator_
best_params = gs_rf.best_params_

r = evaluate("RF + GridSearchCV (best params)", best_rf, X_test, y_test)
r["time"]   = elapsed
r["params"] = best_params
results.append(r)

print(f"  Best params: {best_params}")
print(f"  CV best F1-weighted: {gs_rf.best_score_*100:.2f}%")
print(f"  Test Accuracy: {r['accuracy']*100:.2f}%  |  F1-Weighted: {r['f1_weighted']*100:.2f}%")
print(f"  F1 Hadir: {r['f1_hadir']*100:.2f}%  |  F1 Tidak Hadir: {r['f1_tidak']*100:.2f}%")
print(f"  Time: {elapsed:.1f}s")

# ============================================================
# PHASE 2 — XGBoost
# ============================================================
banner("PHASE 2 — XGBoost")
try:
    from xgboost import XGBClassifier

    # Calculate scale_pos_weight to handle imbalance (ratio majority/minority)
    n_hadir      = int((y_train == 1).sum())
    n_tidak      = int((y_train == 0).sum())
    scale_weight = n_hadir / n_tidak
    print(f"  scale_pos_weight = {scale_weight:.2f} (handles class imbalance)")

    # Basic XGBoost
    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    r = evaluate("XGBoost (default)", xgb, X_test, y_test)
    r["time"] = time.time() - t0
    results.append(r)
    print(f"  Accuracy: {r['accuracy']*100:.2f}%  |  F1-Weighted: {r['f1_weighted']*100:.2f}%")
    print(f"  F1 Hadir: {r['f1_hadir']*100:.2f}%  |  F1 Tidak Hadir: {r['f1_tidak']*100:.2f}%")

    # XGBoost GridSearch
    print("\n  Running XGBoost GridSearch...")
    xgb_param_grid = {
        "n_estimators"  : [100, 200, 300],
        "max_depth"     : [3, 5, 7],
        "learning_rate" : [0.05, 0.1, 0.2],
        "subsample"     : [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    t0 = time.time()
    gs_xgb = GridSearchCV(
        XGBClassifier(
            scale_pos_weight=scale_weight,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            verbosity=0,
        ),
        xgb_param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0,
    )
    gs_xgb.fit(X_train, y_train)
    elapsed = time.time() - t0
    best_xgb = gs_xgb.best_estimator_
    r = evaluate("XGBoost + GridSearchCV", best_xgb, X_test, y_test)
    r["time"]   = elapsed
    r["params"] = gs_xgb.best_params_
    results.append(r)
    print(f"  Best params: {gs_xgb.best_params_}")
    print(f"  CV best F1-weighted: {gs_xgb.best_score_*100:.2f}%")
    print(f"  Test Accuracy: {r['accuracy']*100:.2f}%  |  F1-Weighted: {r['f1_weighted']*100:.2f}%")
    print(f"  F1 Hadir: {r['f1_hadir']*100:.2f}%  |  F1 Tidak Hadir: {r['f1_tidak']*100:.2f}%")
    print(f"  Time: {elapsed:.1f}s")
    xgb_available = True

except ImportError:
    print("  XGBoost tidak terinstal. Jalankan: pip install xgboost")
    print("  Melanjutkan tanpa Phase 2...")
    best_xgb      = None
    xgb_available = False

# ============================================================
# PHASE 3 — SMOTE + Best RF
# ============================================================
banner("PHASE 3 — SMOTE + Random Forest (GridSearch best params)")
try:
    from imblearn.over_sampling import SMOTE

    print(f"  Before SMOTE — Train: Hadir={(y_train==1).sum()}, Tidak Hadir={(y_train==0).sum()}")
    smote = SMOTE(random_state=RANDOM_SEED)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    print(f"  After  SMOTE — Train: Hadir={(y_sm==1).sum()}, Tidak Hadir={(y_sm==0).sum()}")

    # SMOTE + GridSearch best RF params
    t0 = time.time()
    rf_smote = RandomForestClassifier(
        **{k: v for k, v in best_params.items()},
        random_state=RANDOM_SEED,
        class_weight=None,   # SMOTE already balanced — no need for class_weight
    )
    rf_smote.fit(X_sm, y_sm)
    r = evaluate("SMOTE + RF (GridSearch params)", rf_smote, X_test, y_test)
    r["time"] = time.time() - t0
    results.append(r)
    print(f"  Accuracy: {r['accuracy']*100:.2f}%  |  F1-Weighted: {r['f1_weighted']*100:.2f}%")
    print(f"  F1 Hadir: {r['f1_hadir']*100:.2f}%  |  F1 Tidak Hadir: {r['f1_tidak']*100:.2f}%")
    smote_available = True

except ImportError:
    print("  imbalanced-learn tidak terinstal. Jalankan: pip install imbalanced-learn")
    print("  Melanjutkan tanpa Phase 3...")
    smote_available = False

# ============================================================
# COMPARISON TABLE + PICK BEST
# ============================================================
banner("HASIL PERBANDINGAN SEMUA MODEL")

# Pick best by accuracy (primary) then f1_weighted (tiebreak)
best_r = max(results, key=lambda r: (r["accuracy"], r["f1_weighted"]))
for r in results:
    r["best"] = (r is best_r)

print_results(results)

print(f"\n  ✅  Model terbaik: {best_r['name']}")
print(f"      Accuracy     : {best_r['accuracy']*100:.2f}%")
print(f"      F1-Weighted  : {best_r['f1_weighted']*100:.2f}%")
print(f"      F1 Hadir     : {best_r['f1_hadir']*100:.2f}%")
print(f"      F1 Tdk Hadir : {best_r['f1_tidak']*100:.2f}%")
if "params" in best_r:
    print(f"      Best params  : {best_r['params']}")

# ============================================================
# SAVE BEST MODEL
# ============================================================
banner("SIMPAN MODEL TERBAIK")

# Determine which fitted model object to save
model_map = {
    "RF Baseline (n=100, default)"       : rf_base,
    "RF + GridSearchCV (best params)"    : best_rf,
}
if xgb_available:
    model_map["XGBoost (default)"]          = xgb
    model_map["XGBoost + GridSearchCV"]     = best_xgb
if smote_available:
    model_map["SMOTE + RF (GridSearch params)"] = rf_smote

best_model = model_map[best_r["name"]]

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(best_model, OUTPUT_PKL)
print(f"  Saved: {OUTPUT_PKL}")

# Update split_info.json
with open(SPLIT_INFO) as f:
    split = json.load(f)
split["experiment_best_model"] = best_r["name"]
split["experiment_accuracy"]   = round(best_r["accuracy"] * 100, 4)
if "params" in best_r:
    split["best_params"] = best_r["params"]
with open(SPLIT_INFO, "w") as f:
    json.dump(split, f, indent=2)
print(f"  Updated: {SPLIT_INFO}")

print("\n[OK] Selesai. Jalankan 03_evaluate.py untuk update metrics_summary.json\n")
