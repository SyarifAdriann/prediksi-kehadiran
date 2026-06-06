"""
01_preprocess.py
----------------
Tahap KDD: Data Selection, Cleaning, Transformasi, Feature Engineering.

Input  : data/raw/event1_raw.xlsx
         data/raw/event2_raw.xlsx
Output : data/processed/dataset_gabungan.csv

Jalankan dari root folder skripsi/:
    python scripts/01_preprocess.py
"""

import os
import math
import re
import pandas as pd
import numpy as np

# ============================================================
# KONSTANTA
# ============================================================
RANDOM_SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_PROC = os.path.join(ROOT, "data", "processed")

FILE_E1 = os.path.join(DATA_RAW, "event1_raw.xlsx")
FILE_E2 = os.path.join(DATA_RAW, "event2_raw.xlsx")
OUTPUT_CSV = os.path.join(DATA_PROC, "dataset_gabungan.csv")

# Koordinat lokasi event (PT CLN, Halim Perdanakusuma)
HALIM_LAT = -6.2634
HALIM_LON = 106.8910

# Koordinat sentroid kecamatan (untuk koreksi outlier jarak)
KECAMATAN_COORDS = {
    "Kramat Jati":  (-6.2700, 106.8700),
    "Makasar":      (-6.2550, 106.8950),
    "Cipayung":     (-6.3100, 106.9000),
    "Ciracas":      (-6.3300, 106.9100),
    "Cakung":       (-6.2300, 106.9500),
    "Pulo Gadung":  (-6.2000, 106.9000),
    "Jatinegara":   (-6.2150, 106.8700),
    "Matraman":     (-6.2000, 106.8600),
    "Pasar Rebo":   (-6.3350, 106.8800),
    "Tebet":        (-6.2400, 106.8500),
    "Pancoran":     (-6.2550, 106.8400),
    "Jagakarsa":    (-6.3500, 106.8300),
    "Setiabudi":    (-6.2200, 106.8300),
    "Bekasi Barat": (-6.2400, 106.9900),
    "Bekasi Timur": (-6.2350, 107.0100),
    "Pondok Gede":  (-6.2900, 106.9700),
    "Cimanggis":    (-6.3700, 106.9200),
    "Beji":         (-6.3900, 106.8300),
    "Cilincing":    (-6.1100, 106.9300),
    "Pasar Manggis":(-6.2200, 106.8400),
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Hitung jarak (km) antara dua titik koordinat."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def parse_jarak(val):
    """
    Parse string Jarak yang tidak konsisten menjadi float (km).
    Contoh:
      '12 KM'   -> 12.0
      '400 M'   -> 0.4
      '2.5 KM'  -> 2.5
      '13,8 KM' -> 13.8  (koma sebagai desimal)
      '4.5 KM ' -> 4.5   (trailing space)
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    s = str(val).strip().upper().replace(",", ".")
    # Hapus spasi di dalam angka (e.g., "1 2 KM" -> "12 KM")
    s = re.sub(r'(\d)\s+(\d)', r'\1\2', s)
    try:
        if "KM" in s:
            num = float(s.replace("KM", "").strip())
            return round(num, 2)
        elif s.endswith("M"):
            num = float(s.replace("M", "").strip())
            return round(num / 1000.0, 4)
        else:
            # Coba parse langsung sebagai angka
            return round(float(s), 2)
    except (ValueError, AttributeError):
        return None


def standardize_kecamatan(val):
    """Standardisasi nama Kecamatan."""
    if not isinstance(val, str):
        return val
    v = val.strip()
    # Hapus prefix "Kecamatan" / newline
    v = re.sub(r'(?i)kecamatan\s*', '', v).strip()
    v = v.replace("\n", " ").strip()
    # Mapping koreksi
    mapping = {
        "Kramat jati":   "Kramat Jati",
        "kramat jati":   "Kramat Jati",
        "Kramat Jati":   "Kramat Jati",
        "Makassar":      "Makasar",
        "makassar":      "Makasar",
        "makasar":       "Makasar",
        "Makasar":       "Makasar",
        "Ps Rebo":       "Pasar Rebo",
        "Ps. Rebo":      "Pasar Rebo",
        "Pulo gadung":   "Pulo Gadung",
        "pulo gadung":   "Pulo Gadung",
        "Cilincing":     "Cilincing",
    }
    return mapping.get(v, v)


def standardize_kab(val):
    """Standardisasi nama Kabupaten/Kota."""
    if not isinstance(val, str):
        return val
    v = val.strip().lower()
    if re.search(r'jakarta.*(timur|east)', v):
        return "Jakarta Timur"
    if re.search(r'jakarta.*(selatan|south)', v):
        return "Jakarta Selatan"
    if re.search(r'jakarta.*(utara|north)', v):
        return "Jakarta Utara"
    if re.search(r'jakarta.*(barat|west)', v):
        return "Jakarta Barat"
    if re.search(r'jakarta.*(pusat|central)', v):
        return "Jakarta Pusat"
    if v in ("jakarta", "dki jakarta"):
        return "Jakarta Timur"  # default jika tidak ada arah
    if re.search(r'administrasi.*jak(sel|selatan)', v):
        return "Jakarta Selatan"
    if "bekasi" in v:
        return "Bekasi"
    if "depok" in v:
        return "Depok"
    if "bogor" in v:
        return "Bogor"
    if "tangerang" in v:
        return "Tangerang"
    return val.strip().title()


def standardize_prov(val):
    """Standardisasi nama Provinsi."""
    if not isinstance(val, str):
        return val
    v = val.strip().lower()
    if "jakarta" in v:
        return "DKI Jakarta"
    if "jawa barat" in v or "west java" in v:
        return "Jawa Barat"
    if "banten" in v:
        return "Banten"
    if "jawa tengah" in v:
        return "Jawa Tengah"
    if "jawa timur" in v:
        return "Jawa Timur"
    return val.strip().title()


def correct_jarak_outliers(df):
    """
    Koreksi nilai Jarak yang tidak masuk akal:
    Jika jarak tercatat > 3x jarak referensi kecamatan DAN > 10 km lebih besar,
    ganti dengan jarak referensi ± variasi kecil.
    """
    np.random.seed(RANDOM_SEED)
    corrected = []
    for _, row in df.iterrows():
        kec = row.get("Kecamatan", "")
        jarak_km = row.get("_jarak_km_raw", None)
        if kec in KECAMATAN_COORDS and jarak_km is not None:
            ref_lat, ref_lon = KECAMATAN_COORDS[kec]
            ref_km = haversine_km(ref_lat, ref_lon, HALIM_LAT, HALIM_LON)
            # Flagging outlier: jika > 3x expected DAN selisih > 10 km
            if jarak_km > ref_km * 3.0 and jarak_km > ref_km + 10:
                corrected_km = max(0.1, ref_km + np.random.uniform(-0.5, 1.5))
                corrected.append(round(corrected_km, 2))
            else:
                corrected.append(jarak_km)
        else:
            corrected.append(jarak_km)
    return corrected


# ============================================================
# LOAD & CLEAN
# ============================================================

def load_event(filepath, event_id):
    """Load satu file Excel event, bersihkan, dan tambahkan Event_ID."""
    print(f"\n  Loading Event {event_id}: {os.path.basename(filepath)}")
    df = pd.read_excel(filepath, header=2)

    # Rename kolom ke standar (hapus trailing/leading spaces)
    df.columns = df.columns.str.strip()

    expected_cols = [
        "No.", "ID ANGGOTA", "Usia", "Jenis Kelamin",
        "Alamat tempat tinggal", "Desa/Kelurahan", "Kecamatan",
        "Kabupaten/Kota", "Provinsi", "Jarak",
        "Status Pendaftaran", "Status Kehadiran"
    ]
    df = df[expected_cols].copy()
    df = df.drop(columns=["No."])  # nomor urut tidak berguna

    print(f"    Shape: {df.shape}")

    # -------------------------------------------------------
    # Data Cleaning
    # -------------------------------------------------------

    # Standardisasi kolom teks
    df["Kecamatan"] = df["Kecamatan"].apply(standardize_kecamatan)
    df["Kabupaten/Kota"] = df["Kabupaten/Kota"].apply(standardize_kab)
    df["Provinsi"] = df["Provinsi"].apply(standardize_prov)

    # Koreksi typo Jakarta Utata → Jakarta Utara
    df["Kabupaten/Kota"] = df["Kabupaten/Kota"].str.replace(
        "Jakarta Utata", "Jakarta Utara", regex=False
    )

    # Strip whitespace di semua string columns
    str_cols = ["ID ANGGOTA", "Jenis Kelamin", "Alamat tempat tinggal",
                "Desa/Kelurahan", "Kecamatan", "Kabupaten/Kota",
                "Provinsi", "Jarak", "Status Pendaftaran", "Status Kehadiran"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("nan", None)

    # -------------------------------------------------------
    # Parse Jarak → float km
    # -------------------------------------------------------
    df["_jarak_km_raw"] = df["Jarak"].apply(parse_jarak)

    # Koreksi outlier jarak berdasarkan kecamatan
    df["Jarak_km"] = correct_jarak_outliers(df)

    # Hapus kolom sementara
    df = df.drop(columns=["_jarak_km_raw"])

    # Isi Jarak_km yang masih None dengan median kecamatan jika ada
    median_jarak = df["Jarak_km"].median()
    df["Jarak_km"] = df["Jarak_km"].fillna(median_jarak)
    df["Jarak_km"] = df["Jarak_km"].round(2)

    # -------------------------------------------------------
    # Event_ID
    # -------------------------------------------------------
    df["Event_ID"] = event_id

    print(f"    Jarak_km range: {df['Jarak_km'].min():.2f} – {df['Jarak_km'].max():.2f} km")
    print(f"    Status Kehadiran: {df['Status Kehadiran'].value_counts().to_dict()}")
    print(f"    Null counts:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")

    return df


# ============================================================
# MERGE & FEATURE ENGINEERING
# ============================================================

def engineer_hadir_event_sebelumnya(df):
    """
    Buat fitur 'hadir_event_sebelumnya':
    - Untuk baris Event 1: selalu 0 (belum ada riwayat)
    - Untuk baris Event 2: 1 jika peserta yang sama hadir di Event 1, 0 jika tidak hadir atau tidak ada
    Matching berdasarkan nama (ID ANGGOTA), case-insensitive, strip.
    """
    df = df.copy()

    # Buat lookup: nama → Status Kehadiran di Event 1
    df1_rows = df[df["Event_ID"] == 1].copy()
    df1_rows["_nama_key"] = df1_rows["ID ANGGOTA"].str.strip().str.lower()
    kehadiran_e1 = dict(
        zip(df1_rows["_nama_key"],
            df1_rows["Status Kehadiran"])
    )

    hadir_sblm = []
    for _, row in df.iterrows():
        if row["Event_ID"] == 1:
            hadir_sblm.append(0)
        else:
            key = str(row["ID ANGGOTA"]).strip().lower()
            sk_e1 = kehadiran_e1.get(key, None)
            if sk_e1 == "Hadir":
                hadir_sblm.append(1)
            else:
                hadir_sblm.append(0)

    df["hadir_event_sebelumnya"] = hadir_sblm

    # Statistik
    e2 = df[df["Event_ID"] == 2]
    overlap_count = (e2["hadir_event_sebelumnya"] == 1).sum()
    print(f"    hadir_event_sebelumnya=1 di Event 2: {overlap_count} peserta")

    return df.drop(columns=["_nama_key"], errors="ignore")


# ============================================================
# ENCODING
# ============================================================

def encode_features(df):
    """
    Label encoding fitur kategorik:
      Jenis Kelamin:     Perempuan=0, Laki-laki=1
      Status Pendaftaran: Terdaftar=1, Tidak Terdaftar=0
      Status Kehadiran (target): Hadir=1, Tidak Hadir=0
    """
    df = df.copy()

    # Jenis Kelamin
    jk_map = {"Perempuan": 0, "Laki - Laki": 1, "Laki-Laki": 1, "Laki - laki": 1}
    df["Jenis_Kelamin"] = df["Jenis Kelamin"].map(jk_map)
    unmapped_jk = df[df["Jenis_Kelamin"].isna()]["Jenis Kelamin"].unique()
    if len(unmapped_jk) > 0:
        print(f"    [WARNING] Jenis Kelamin tidak dikenali: {unmapped_jk}")
    df["Jenis_Kelamin"] = df["Jenis_Kelamin"].fillna(0).astype(int)

    # Status Pendaftaran
    sp_map = {"Terdaftar": 1, "Tidak Terdaftar": 0}
    df["Status_Pendaftaran"] = df["Status Pendaftaran"].map(sp_map)
    unmapped_sp = df[df["Status_Pendaftaran"].isna()]["Status Pendaftaran"].unique()
    if len(unmapped_sp) > 0:
        print(f"    [WARNING] Status Pendaftaran tidak dikenali: {unmapped_sp}")
    df["Status_Pendaftaran"] = df["Status_Pendaftaran"].fillna(1).astype(int)

    # Status Kehadiran (target)
    sk_map = {"Hadir": 1, "Tidak Hadir": 0}
    df["Status_Kehadiran"] = df["Status Kehadiran"].map(sk_map)
    unmapped_sk = df[df["Status_Kehadiran"].isna()]["Status Kehadiran"].unique()
    if len(unmapped_sk) > 0:
        print(f"    [WARNING] Status Kehadiran tidak dikenali: {unmapped_sk}")
    # Drop baris dengan target kosong (seharusnya tidak ada)
    before = len(df)
    df = df.dropna(subset=["Status_Kehadiran"])
    if len(df) < before:
        print(f"    [WARNING] {before - len(df)} baris dihapus karena Status Kehadiran kosong")
    df["Status_Kehadiran"] = df["Status_Kehadiran"].astype(int)

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("01_preprocess.py — KDD Preprocessing Pipeline")
    print("=" * 60)

    os.makedirs(DATA_PROC, exist_ok=True)

    # ----------------------------------------------------------
    # Load & clean kedua event
    # ----------------------------------------------------------
    print("\n[STEP 1] Load & Clean Data")
    df1 = load_event(FILE_E1, event_id=1)
    df2 = load_event(FILE_E2, event_id=2)

    # ----------------------------------------------------------
    # Merge
    # ----------------------------------------------------------
    print("\n[STEP 2] Merge Dataset")
    df = pd.concat([df1, df2], ignore_index=True)
    print(f"  Dataset gabungan: {df.shape}")

    # ----------------------------------------------------------
    # Feature Engineering: hadir_event_sebelumnya
    # ----------------------------------------------------------
    print("\n[STEP 3] Feature Engineering: hadir_event_sebelumnya")
    df = engineer_hadir_event_sebelumnya(df)

    # ----------------------------------------------------------
    # Encoding
    # ----------------------------------------------------------
    print("\n[STEP 4] Encoding Kategorik")
    df = encode_features(df)

    # ----------------------------------------------------------
    # Pilih & urutkan kolom final
    # ----------------------------------------------------------
    print("\n[STEP 5] Seleksi Kolom Final")

    # Kolom referensi (tidak masuk model, dipertahankan untuk traceability)
    ref_cols = ["ID ANGGOTA", "Jenis Kelamin", "Status Pendaftaran",
                "Status Kehadiran", "Jarak", "Kecamatan",
                "Kabupaten/Kota", "Provinsi", "Desa/Kelurahan",
                "Alamat tempat tinggal"]

    # Kolom model
    model_cols = ["Jenis_Kelamin", "Usia", "Jarak_km",
                  "Status_Pendaftaran", "Event_ID",
                  "hadir_event_sebelumnya", "Status_Kehadiran"]

    final_cols = ref_cols + model_cols
    # Hanya ambil yang ada
    final_cols = [c for c in final_cols if c in df.columns]
    df_final = df[final_cols].copy()

    # ----------------------------------------------------------
    # Simpan
    # ----------------------------------------------------------
    print("\n[STEP 6] Simpan Dataset")
    df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"  Saved: {OUTPUT_CSV}")

    # ----------------------------------------------------------
    # Ringkasan Akhir
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("RINGKASAN PREPROCESSING")
    print("=" * 60)
    print(f"  Total baris    : {len(df_final)}")
    print(f"  Total kolom    : {len(df_final.columns)}")
    print(f"  Fitur model    : Jenis_Kelamin, Usia, Jarak_km,")
    print(f"                   Status_Pendaftaran, Event_ID, hadir_event_sebelumnya")
    print(f"  Target         : Status_Kehadiran")
    print(f"\n  Distribusi Target:")
    vc = df_final["Status_Kehadiran"].value_counts()
    for val, cnt in vc.items():
        label = "Hadir" if val == 1 else "Tidak Hadir"
        print(f"    {label} ({val}): {cnt} ({cnt/len(df_final)*100:.1f}%)")
    print(f"\n  Distribusi per Event:")
    for ev in [1, 2]:
        sub = df_final[df_final["Event_ID"] == ev]
        hadir = (sub["Status_Kehadiran"] == 1).sum()
        print(f"    Event {ev}: {len(sub)} rows | Hadir: {hadir} ({hadir/len(sub)*100:.1f}%)")
    print(f"\n  Statistik Usia:")
    print(f"    Mean={df_final['Usia'].mean():.1f}, Std={df_final['Usia'].std():.1f}, "
          f"Min={df_final['Usia'].min()}, Max={df_final['Usia'].max()}")
    print(f"\n  Statistik Jarak_km:")
    print(f"    Mean={df_final['Jarak_km'].mean():.2f}, Std={df_final['Jarak_km'].std():.2f}, "
          f"Min={df_final['Jarak_km'].min():.2f}, Max={df_final['Jarak_km'].max():.2f}")
    print(f"\n  hadir_event_sebelumnya:")
    hsb = df_final["hadir_event_sebelumnya"].value_counts()
    for v, c in hsb.items():
        print(f"    = {v}: {c} baris")

    print("\n[OK] Preprocessing selesai.\n")


if __name__ == "__main__":
    main()
