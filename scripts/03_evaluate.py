"""
03_evaluate.py
--------------
Tahap KDD: Evaluasi Model dan Visualisasi.

Input  : models/random_forest_model.pkl
         models/split_info.json
         data/processed/dataset_gabungan.csv
Output : outputs/metrics_summary.json
         outputs/confusion_matrix.png
         outputs/feature_importance.png
         outputs/distribusi_kehadiran.png
         outputs/distribusi_usia.png
         outputs/distribusi_jarak.png

Jalankan dari root folder skripsi/:
    python scripts/03_evaluate.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ============================================================
# KONSTANTA
# ============================================================
RANDOM_SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROC = os.path.join(ROOT, "data", "processed")
MODEL_DIR = os.path.join(ROOT, "models")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

INPUT_CSV = os.path.join(DATA_PROC, "dataset_gabungan.csv")
MODEL_PKL = os.path.join(MODEL_DIR, "random_forest_model.pkl")
SPLIT_JSON = os.path.join(MODEL_DIR, "split_info.json")
METRICS_JSON = os.path.join(OUTPUT_DIR, "metrics_summary.json")

# ---- Gaya tampilan grafik ----
MONO_DARK = "#1a1a1a"
MONO_MID = "#555555"
MONO_LIGHT = "#aaaaaa"
MONO_PALE = "#e8e8e8"
WHITE = "#ffffff"
ACCENT = "#333333"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "axes.edgecolor": MONO_LIGHT,
    "axes.labelcolor": MONO_DARK,
    "xtick.color": MONO_MID,
    "ytick.color": MONO_MID,
    "text.color": MONO_DARK,
})


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("03_evaluate.py — Evaluasi Model & Visualisasi")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------------------------------------------
    # Load
    # ----------------------------------------------------------
    print("\n[STEP 1] Load Model & Dataset")
    model = joblib.load(MODEL_PKL)
    print(f"  Model loaded: {MODEL_PKL}")

    df = pd.read_csv(INPUT_CSV)
    print(f"  Dataset shape: {df.shape}")

    with open(SPLIT_JSON) as f:
        split_info = json.load(f)

    feature_cols = split_info["feature_cols"]
    target_col = split_info["target_col"]
    test_size = split_info["test_size"]

    X = df[feature_cols]
    y = df[target_col]

    # Rekonstruksi test set (sama persis dengan yang digunakan di 02_train.py)
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # ----------------------------------------------------------
    # Prediksi
    # ----------------------------------------------------------
    print("\n[STEP 2] Prediksi pada Test Set")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # ----------------------------------------------------------
    # Metrik Evaluasi
    # ----------------------------------------------------------
    print("\n[STEP 3] Hitung Metrik Evaluasi")

    acc = accuracy_score(y_test, y_pred)
    prec_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Per-kelas
    prec_hadir = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_hadir = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_hadir = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    prec_tidak = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec_tidak = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_tidak = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"  Accuracy  (weighted): {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision (weighted): {prec_w:.4f} ({prec_w*100:.2f}%)")
    print(f"  Recall    (weighted): {rec_w:.4f} ({rec_w*100:.2f}%)")
    print(f"  F1-Score  (weighted): {f1_w:.4f} ({f1_w*100:.2f}%)")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Tidak Hadir", "Hadir"],
                                zero_division=0))

    # Kepentingan fitur (feature importance)
    fi = model.feature_importances_
    fi_dict = dict(zip(feature_cols, fi.tolist()))
    fi_sorted = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)
    print(f"  Feature Importance:")
    for feat, imp in fi_sorted:
        print(f"    {feat}: {imp:.4f}")

    # ----------------------------------------------------------
    # Simpan metrics_summary.json
    # ----------------------------------------------------------
    print("\n[STEP 4] Simpan metrics_summary.json")
    metrics = {
        "n_test": int(len(y_test)),
        "n_train": split_info["n_train"],
        "n_total": int(len(df)),
        "accuracy": round(float(acc), 4),
        "accuracy_pct": round(float(acc) * 100, 2),
        "precision_weighted": round(float(prec_w), 4),
        "recall_weighted": round(float(rec_w), 4),
        "f1_weighted": round(float(f1_w), 4),
        "per_class": {
            "Hadir": {
                "precision": round(float(prec_hadir), 4),
                "recall": round(float(rec_hadir), 4),
                "f1": round(float(f1_hadir), 4),
                "support": int((y_test == 1).sum()),
            },
            "Tidak Hadir": {
                "precision": round(float(prec_tidak), 4),
                "recall": round(float(rec_tidak), 4),
                "f1": round(float(f1_tidak), 4),
                "support": int((y_test == 0).sum()),
            },
        },
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
        },
        "feature_importance": fi_dict,
        "feature_importance_sorted": [
            {"feature": k, "importance": round(v, 4)} for k, v in fi_sorted
        ],
        "model_params": {
            "n_estimators": model.n_estimators,
            "random_state": RANDOM_SEED,
            "class_weight": "balanced",
            "test_size": test_size,
        },
        "distribusi_dataset": {
            "total": int(len(df)),
            "hadir": int((df[target_col] == 1).sum()),
            "tidak_hadir": int((df[target_col] == 0).sum()),
            "hadir_pct": round((df[target_col] == 1).mean() * 100, 2),
        },
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {METRICS_JSON}")

    # ──────────────────────────────────────────────────────────
    # Visualisasi 1: Confusion Matrix
    # ──────────────────────────────────────────────────────────
    print("\n[STEP 5] Membuat Grafik...")

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Tidak Hadir", "Hadir"]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greys",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor=MONO_PALE,
        annot_kws={"size": 16, "weight": "bold"},
        ax=ax
    )
    ax.set_title("Confusion Matrix\nPrediksi Kehadiran Anggota", pad=15)
    ax.set_xlabel("Prediksi", labelpad=10)
    ax.set_ylabel("Aktual", labelpad=10)
    plt.tight_layout()
    out_cm = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(out_cm, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_cm}")

    # ──────────────────────────────────────────────────────────
    # Visualisasi 2: Kepentingan Fitur (Feature Importance)
    # ──────────────────────────────────────────────────────────
    feat_labels = {
        "Jenis_Kelamin": "Jenis Kelamin",
        "Usia": "Usia",
        "Jarak_km": "Jarak (km)",
        "Status_Pendaftaran": "Status Pendaftaran",
        "Event_ID": "Event ID",
        "hadir_event_sebelumnya": "Hadir Event Sebelumnya",
    }
    feats = [feat_labels.get(k, k) for k, _ in fi_sorted]
    imps = [v for _, v in fi_sorted]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(feats[::-1], imps[::-1], color=[MONO_MID] * len(feats), edgecolor=WHITE)
    # Highlight batang teratas
    bars[-1].set_color(MONO_DARK)
    ax.set_xlabel("Tingkat Kepentingan (Importance)", labelpad=10)
    ax.set_title("Feature Importance\nAlgoritma Random Forest", pad=15)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    for bar, val in zip(bars, imps[::-1]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=9, color=MONO_DARK)
    ax.set_xlim(0, max(imps) * 1.25)
    plt.tight_layout()
    out_fi = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(out_fi, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_fi}")

    # ──────────────────────────────────────────────────────────
    # Visualisasi 3: Distribusi Kehadiran per Event
    # ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for i, ev_id in enumerate([1, 2]):
        ev_df = df[df["Event_ID"] == ev_id]
        counts = ev_df["Status_Kehadiran"].value_counts().sort_index()
        labels_ev = ["Tidak Hadir", "Hadir"]
        vals = [counts.get(0, 0), counts.get(1, 0)]
        colors_ev = [MONO_PALE, MONO_DARK]

        bars_ev = axes[i].bar(labels_ev, vals, color=colors_ev, width=0.5, edgecolor=WHITE)
        axes[i].set_title(f"Event {ev_id} — {['09 Nov', '16 Nov'][i]} 2025", pad=12)
        axes[i].set_ylabel("Jumlah Peserta" if i == 0 else "")
        axes[i].set_ylim(0, max(vals) * 1.2)
        for bar, val in zip(bars_ev, vals):
            axes[i].text(bar.get_x() + bar.get_width() / 2, val + 1,
                         str(val), ha="center", fontsize=12, fontweight="bold")

    fig.suptitle("Distribusi Status Kehadiran per Event", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_dk = os.path.join(OUTPUT_DIR, "distribusi_kehadiran.png")
    plt.savefig(out_dk, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dk}")

    # ──────────────────────────────────────────────────────────
    # Visualisasi 4: Distribusi Usia
    # ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["Usia"], bins=15, color=MONO_MID, edgecolor=WHITE, linewidth=0.8)
    ax.axvline(df["Usia"].mean(), color=MONO_DARK, linestyle="--", linewidth=1.5,
               label=f"Rata-rata: {df['Usia'].mean():.1f} tahun")
    ax.set_xlabel("Usia (tahun)", labelpad=10)
    ax.set_ylabel("Frekuensi", labelpad=10)
    ax.set_title("Distribusi Usia Peserta", pad=15)
    ax.legend(frameon=False)
    plt.tight_layout()
    out_usia = os.path.join(OUTPUT_DIR, "distribusi_usia.png")
    plt.savefig(out_usia, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_usia}")

    # ──────────────────────────────────────────────────────────
    # Visualisasi 5: Distribusi Jarak
    # ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Di-cap pada 25km agar grafik lebih mudah dibaca (outlier jarak jauh tidak mendominasi)
    jarak_plot = df["Jarak_km"].clip(upper=25)
    ax.hist(jarak_plot, bins=20, color=MONO_MID, edgecolor=WHITE, linewidth=0.8)
    ax.axvline(df["Jarak_km"].median(), color=MONO_DARK, linestyle="--", linewidth=1.5,
               label=f"Median: {df['Jarak_km'].median():.1f} km")
    ax.set_xlabel("Jarak ke Lokasi Event (km, di-cap pada 25 km)", labelpad=10)
    ax.set_ylabel("Frekuensi", labelpad=10)
    ax.set_title("Distribusi Jarak Tempat Tinggal Peserta", pad=15)
    ax.legend(frameon=False)
    plt.tight_layout()
    out_jarak = os.path.join(OUTPUT_DIR, "distribusi_jarak.png")
    plt.savefig(out_jarak, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_jarak}")

    # ──────────────────────────────────────────────────────────
    # Ringkasan
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RINGKASAN EVALUASI MODEL")
    print("=" * 60)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec_w*100:.2f}% (weighted)")
    print(f"  Recall    : {rec_w*100:.2f}% (weighted)")
    print(f"  F1-Score  : {f1_w*100:.2f}% (weighted)")
    print(f"\n  Per Kelas:")
    print(f"    Hadir      — Precision: {prec_hadir*100:.2f}% | Recall: {rec_hadir*100:.2f}% | F1: {f1_hadir*100:.2f}%")
    print(f"    Tidak Hadir— Precision: {prec_tidak*100:.2f}% | Recall: {rec_tidak*100:.2f}% | F1: {f1_tidak*100:.2f}%")
    print(f"\n  Feature Importance (top):")
    for feat, imp in fi_sorted[:3]:
        print(f"    {feat_labels.get(feat, feat)}: {imp*100:.2f}%")
    print("\n[OK] Evaluasi selesai. Semua output tersimpan di outputs/\n")


if __name__ == "__main__":
    main()
