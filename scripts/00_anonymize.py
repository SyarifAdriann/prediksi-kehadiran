"""
00_anonymize.py
---------------
Anonymisasi kolom 'ID ANGGOTA' (berisi nama asli) pada kedua file Excel mentah.

Alur kerja:
  1. Baca semua nama unik dari event1_raw.xlsx & event2_raw.xlsx
  2. Sort alfabetis (case-insensitive) → tetapkan kode P001, P002, ...
  3. Simpan tabel mapping ke data/processed/anon_mapping.csv
  4. Timpa event1_raw.xlsx & event2_raw.xlsx dengan kode anonim
     (semua kolom lain TIDAK diubah)

Catatan:
  - Matching dilakukan case-insensitive + strip whitespace,
    persis seperti yang dilakukan oleh 01_preprocess.py.
  - Nama yang persis sama di Event 1 dan Event 2 mendapat kode yang sama,
    sehingga fitur hadir_event_sebelumnya tetap terhitung benar.

Jalankan dari root folder skripsi/:
    python scripts/00_anonymize.py
"""

import os
import pandas as pd

RANDOM_SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW   = os.path.join(ROOT, "data", "raw")
DATA_PROC  = os.path.join(ROOT, "data", "processed")

FILE_E1 = os.path.join(DATA_RAW, "event1_raw.xlsx")
FILE_E2 = os.path.join(DATA_RAW, "event2_raw.xlsx")
MAPPING_CSV = os.path.join(DATA_PROC, "anon_mapping.csv")

HEADER_ROW = 2   # Baris ke-2 (0-indexed) berisi nama kolom, sama dengan 01_preprocess.py


def load_raw(filepath):
    """Muat file Excel mentah dengan mempertahankan seluruh struktur file."""
    # Baca file penuh tanpa melewati baris apa pun
    df_full = pd.read_excel(filepath, header=None)
    # Header kolom sebenarnya ada di baris HEADER_ROW
    return df_full


def get_names_from_file(filepath):
    """Ambil semua nilai kolom 'ID ANGGOTA' dari file Excel mentah."""
    df = pd.read_excel(filepath, header=HEADER_ROW)
    df.columns = df.columns.str.strip()
    # Hapus baris yang ID ANGGOTA-nya kosong atau NaN
    names = df["ID ANGGOTA"].dropna().astype(str).str.strip()
    names = names[names.str.len() > 0]
    return names.tolist()


def build_mapping(names_e1, names_e2):
    """
    Bangun tabel mapping nama → kode secara deterministik.

    Kunci dinormalisasi (huruf kecil + strip) agar pencocokan tidak
    bergantung pada huruf besar/kecil (case-insensitive).
    Kode berupa bilangan bulat berurutan dengan zero-padding: P001, P002, ...
    Urutan ditentukan secara alfabetis dari kumpulan nama gabungan kedua event.
    """
    # Kumpulkan semua nama unik (pertahankan huruf asli untuk tampilan)
    seen_lower = {}   # kunci_lowercase → penulisan_asli_pertama_ditemukan

    for name in names_e1 + names_e2:
        key = name.strip().lower()
        if key not in seen_lower:
            seen_lower[key] = name.strip()

    # Urutkan berdasarkan kunci yang dinormalisasi (alfabetis, case-insensitive)
    sorted_keys = sorted(seen_lower.keys())

    width = max(3, len(str(len(sorted_keys))))   # minimal 3 digit → P001

    mapping_lower = {}    # kunci_normal → kode   (untuk pencarian cepat)
    mapping_display = {}  # penulisan_asli → kode  (untuk CSV)

    for idx, key in enumerate(sorted_keys, start=1):
        code = f"P{str(idx).zfill(width)}"
        orig = seen_lower[key]
        mapping_lower[key] = code
        mapping_display[orig] = code

    return mapping_lower, mapping_display


def anonymize_file(filepath, mapping_lower):
    """
    Ganti kolom 'ID ANGGOTA' dalam file Excel dengan kode anonim.
    Semua kolom dan baris lain dipertahankan persis seperti aslinya.
    Mengembalikan DataFrame mentah (tanpa parsing header) yang sudah
    dimodifikasi, siap ditulis ulang ke Excel.
    """
    # Baca file Excel penuh tanpa parsing header
    raw = pd.read_excel(filepath, header=None)

    # Data sebenarnya dimulai di baris HEADER_ROW (0-indexed)
    # Baris 0 .. HEADER_ROW-1 adalah baris judul/metadata → biarkan
    # Baris HEADER_ROW adalah baris nama kolom
    # Baris HEADER_ROW+1 .. akhir adalah data

    header_row_idx = HEADER_ROW
    col_header = raw.iloc[header_row_idx].tolist()

    # Cari indeks kolom 'ID ANGGOTA'
    id_col_idx = None
    for i, h in enumerate(col_header):
        if isinstance(h, str) and h.strip() == "ID ANGGOTA":
            id_col_idx = i
            break

    if id_col_idx is None:
        raise ValueError(f"Kolom 'ID ANGGOTA' tidak ditemukan di {filepath}. Header: {col_header}")

    print(f"    Kolom 'ID ANGGOTA' ada di indeks {id_col_idx}")

    # Ganti nilai: iterasi baris data (setelah baris header)
    replaced_count = 0
    not_found = []

    for row_idx in range(header_row_idx + 1, len(raw)):
        cell = raw.iat[row_idx, id_col_idx]
        if pd.isna(cell) or str(cell).strip() == "":
            continue  # biarkan sel kosong / NaN tidak diubah

        original = str(cell).strip()
        key = original.lower()
        code = mapping_lower.get(key)

        if code is None:
            not_found.append(original)
            # Biarkan tidak berubah jika tidak ditemukan di mapping (seharusnya tidak terjadi)
        else:
            raw.iat[row_idx, id_col_idx] = code
            replaced_count += 1

    if not_found:
        print(f"    [PERINGATAN] {len(not_found)} nama TIDAK ditemukan di mapping: {set(not_found)}")

    print(f"    Berhasil mengganti {replaced_count} sel nama menjadi kode anonim")
    return raw


def save_df_to_excel(df, filepath):
    """Tulis DataFrame kembali ke Excel tanpa index atau header (format mentah dipertahankan)."""
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, header=False)
    print(f"    Saved: {filepath}")


def main():
    print("=" * 60)
    print("00_anonymize.py — Anonymisasi ID ANGGOTA")
    print("=" * 60)

    os.makedirs(DATA_PROC, exist_ok=True)

    # ── Langkah 1: Kumpulkan semua nama dari kedua event ────────
    print("\n[STEP 1] Membaca nama dari kedua file Excel...")
    names_e1 = get_names_from_file(FILE_E1)
    names_e2 = get_names_from_file(FILE_E2)
    print(f"    Event 1: {len(names_e1)} baris nama")
    print(f"    Event 2: {len(names_e2)} baris nama")

    # ── Langkah 2: Bangun tabel mapping ─────────────────────────
    print("\n[STEP 2] Membangun tabel mapping nama → kode...")
    mapping_lower, mapping_display = build_mapping(names_e1, names_e2)
    print(f"    Total nama unik: {len(mapping_lower)}")
    print(f"    Contoh mapping:")
    for orig, code in list(mapping_display.items())[:5]:
        print(f"      '{orig}' → {code}")
    print(f"      ...")

    # ── Langkah 3: Simpan mapping ke CSV ───────────────────────
    print("\n[STEP 3] Menyimpan tabel mapping...")
    mapping_rows = [{"nama_asli": orig, "kode_anonim": code}
                    for orig, code in sorted(mapping_display.items())]
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(MAPPING_CSV, index=False, encoding="utf-8-sig")
    print(f"    Saved: {MAPPING_CSV} ({len(mapping_df)} baris)")

    # ── Langkah 4: Anonimkan Event 1 ────────────────────────────
    print("\n[STEP 4] Anonymisasi event1_raw.xlsx...")
    raw_e1 = anonymize_file(FILE_E1, mapping_lower)
    save_df_to_excel(raw_e1, FILE_E1)

    # ── Langkah 5: Anonimkan Event 2 ────────────────────────────
    print("\n[STEP 5] Anonymisasi event2_raw.xlsx...")
    raw_e2 = anonymize_file(FILE_E2, mapping_lower)
    save_df_to_excel(raw_e2, FILE_E2)

    # ── Langkah 6: Verifikasi konsistensi lintas event ───────────
    print("\n[STEP 6] Verifikasi konsistensi kode lintas event...")
    anon_e1 = pd.read_excel(FILE_E1, header=HEADER_ROW)
    anon_e2 = pd.read_excel(FILE_E2, header=HEADER_ROW)
    anon_e1.columns = anon_e1.columns.str.strip()
    anon_e2.columns = anon_e2.columns.str.strip()

    codes_e1 = set(anon_e1["ID ANGGOTA"].dropna().astype(str).str.strip())
    codes_e2 = set(anon_e2["ID ANGGOTA"].dropna().astype(str).str.strip())
    overlap = codes_e1 & codes_e2

    print(f"    Kode unik di Event 1 : {len(codes_e1)}")
    print(f"    Kode unik di Event 2 : {len(codes_e2)}")
    print(f"    Kode yang overlap    : {len(overlap)} peserta di kedua event")
    print(f"    Semua kode valid (P-format): "
          f"{all(c.startswith('P') for c in codes_e1 | codes_e2)}")

    # Pastikan tidak ada nama asli yang tersisa
    all_codes = codes_e1 | codes_e2
    any_real_name = any(not c.startswith("P") for c in all_codes)
    if any_real_name:
        bad = [c for c in all_codes if not c.startswith("P")]
        print(f"    [WARNING] Ada nilai non-kode ditemukan: {bad[:10]}")
    else:
        print(f"    [OK] Tidak ada nama asli tersisa di kedua file Excel.")

    print("\n[OK] Anonymisasi selesai.\n")
    print("Langkah selanjutnya:")
    print("  python scripts/01_preprocess.py")
    print("  python scripts/02_train.py")
    print("  python scripts/03_evaluate.py")


if __name__ == "__main__":
    main()
