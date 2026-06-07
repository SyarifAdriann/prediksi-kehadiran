"""
04b_save_xgboost.py
--------------------
Melatih XGBoost dengan best params dari eksperimen dan
menyimpannya sebagai model produksi (menggantikan RF).

Jalankan dari root folder skripsi/:
    python scripts/04b_save_xgboost.py
"""

import os, json, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

RANDOM_SEED  = 42
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV    = os.path.join(ROOT, "data", "processed", "dataset_gabungan.csv")
OUTPUT_PKL   = os.path.join(ROOT, "models", "random_forest_model.pkl")   # same path → app.py unchanged
SPLIT_INFO   = os.path.join(ROOT, "models", "split_info.json")

FEATURE_COLS = [
    "Jenis_Kelamin", "Usia", "Jarak_km",
    "Status_Pendaftaran", "Event_ID", "hadir_event_sebelumnya",
]
TARGET_COL = "Status_Kehadiran"

print("=" * 60)
print("  Saving Best Model: XGBoost (tuned params)")
print("=" * 60)

df = pd.read_csv(INPUT_CSV)
X  = df[FEATURE_COLS]
y  = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)

# Best params from GridSearch experiment
n_hadir      = int((y_train == 1).sum())
n_tidak      = int((y_train == 0).sum())
scale_weight = n_hadir / n_tidak

model = XGBClassifier(
    colsample_bytree = 0.8,
    learning_rate    = 0.2,
    max_depth        = 3,
    n_estimators     = 300,
    subsample        = 1.0,
    scale_pos_weight = scale_weight,
    random_state     = RANDOM_SEED,
    eval_metric      = "logloss",
    verbosity        = 0,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
f1_w   = f1_score(y_test, y_pred, average="weighted")

print(f"\n  Test Accuracy     : {acc*100:.2f}%")
print(f"  F1-Weighted       : {f1_w*100:.2f}%")
print(f"  scale_pos_weight  : {scale_weight:.2f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Tidak Hadir','Hadir'])}")

# Save model
joblib.dump(model, OUTPUT_PKL)
print(f"  Saved: {OUTPUT_PKL}")

# Update split_info
with open(SPLIT_INFO) as f:
    info = json.load(f)
info["model_type"]           = "XGBClassifier"
info["experiment_best_model"] = "XGBoost + GridSearchCV"
info["experiment_accuracy"]   = round(acc * 100, 4)
info["best_params"] = {
    "colsample_bytree": 0.8,
    "learning_rate"   : 0.2,
    "max_depth"       : 3,
    "n_estimators"    : 300,
    "subsample"       : 1.0,
    "scale_pos_weight": round(scale_weight, 4),
}
with open(SPLIT_INFO, "w") as f:
    json.dump(info, f, indent=2)
print(f"  Updated: {SPLIT_INFO}")

print("\n[OK] Jalankan 03_evaluate.py untuk update metrics_summary.json\n")
