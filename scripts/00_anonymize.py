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

Jalankan dari root folder skripsi v2/:
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

HEADER_ROW = 2   # 0-indexed row that contains column names (same as 01_preprocess.py)


def load_raw(filepath):
    """Load raw Excel preserving the full file structure."""
    # Read full file without skipping rows (to detect skip rows later)
    df_full = pd.read_excel(filepath, header=None)
    # The actual header is at row index HEADER_ROW
    return df_full


def get_names_from_file(filepath):
    """Extract all 'ID ANGGOTA' values from a raw Excel file."""
    df = pd.read_excel(filepath, header=HEADER_ROW)
    df.columns = df.columns.str.strip()
    # Drop rows where ID ANGGOTA is entirely NaN / not a valid string
    names = df["ID ANGGOTA"].dropna().astype(str).str.strip()
    names = names[names.str.len() > 0]
    return names.tolist()


def build_mapping(names_e1, names_e2):
    """
    Build a deterministic name → code mapping.

    Keys are normalised (lowercase + stripped) for case-insensitive matching.
    Codes are zero-padded sequential integers: P001, P002, ...
    Alphabetical order across the combined unique-name pool.
    """
    # Collect all unique raw names (preserve original casing for display)
    seen_lower = {}   # lower_key → first_seen_original_casing

    for name in names_e1 + names_e2:
        key = name.strip().lower()
        if key not in seen_lower:
            seen_lower[key] = name.strip()

    # Sort by normalised key (alphabetical, case-insensitive)
    sorted_keys = sorted(seen_lower.keys())

    width = max(3, len(str(len(sorted_keys))))   # at least 3 digits → P001

    mapping_lower = {}    # normalised → code   (used for fast lookup)
    mapping_display = {}  # original casing → code  (for CSV)

    for idx, key in enumerate(sorted_keys, start=1):
        code = f"P{str(idx).zfill(width)}"
        orig = seen_lower[key]
        mapping_lower[key] = code
        mapping_display[orig] = code

    return mapping_lower, mapping_display


def anonymize_file(filepath, mapping_lower):
    """
    Replace 'ID ANGGOTA' column in an Excel file with anonymised codes.
    All other columns and rows are preserved exactly.
    Returns a pandas ExcelWriter-compatible DataFrame list:
    (header_rows_df, data_df) so the header rows above the table are kept.
    """
    # Read the full raw file as a plain dataframe (no header parsing)
    raw = pd.read_excel(filepath, header=None)

    # The actual data starts at row HEADER_ROW (0-indexed)
    # Rows 0 .. HEADER_ROW-1 are metadata/title rows → keep as-is
    # Row HEADER_ROW is the column header row
    # Rows HEADER_ROW+1 .. end are data

    header_row_idx = HEADER_ROW
    col_header = raw.iloc[header_row_idx].tolist()

    # Find the column index for 'ID ANGGOTA'
    id_col_idx = None
    for i, h in enumerate(col_header):
        if isinstance(h, str) and h.strip() == "ID ANGGOTA":
            id_col_idx = i
            break

    if id_col_idx is None:
        raise ValueError(f"'ID ANGGOTA' column not found in {filepath}. Headers: {col_header}")

    print(f"    'ID ANGGOTA' is at column index {id_col_idx}")

    # Build replacement: iterate data rows (after header row)
    replaced_count = 0
    not_found = []

    for row_idx in range(header_row_idx + 1, len(raw)):
        cell = raw.iat[row_idx, id_col_idx]
        if pd.isna(cell) or str(cell).strip() == "":
            continue  # leave blank / NaN cells untouched

        original = str(cell).strip()
        key = original.lower()
        code = mapping_lower.get(key)

        if code is None:
            not_found.append(original)
            # Leave unchanged if not in mapping (shouldn't happen)
        else:
            raw.iat[row_idx, id_col_idx] = code
            replaced_count += 1

    if not_found:
        print(f"    [WARNING] {len(not_found)} names NOT found in mapping: {set(not_found)}")

    print(f"    Replaced {replaced_count} name cells → anonymous codes")
    return raw


def save_df_to_excel(df, filepath):
    """Write dataframe back to Excel without index or header (raw format preserved)."""
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, header=False)
    print(f"    Saved: {filepath}")


def main():
    print("=" * 60)
    print("00_anonymize.py — Anonymisasi ID ANGGOTA")
    print("=" * 60)

    os.makedirs(DATA_PROC, exist_ok=True)

    # ── Step 1: Collect all names from both events ────────────
    print("\n[STEP 1] Membaca nama dari kedua file Excel...")
    names_e1 = get_names_from_file(FILE_E1)
    names_e2 = get_names_from_file(FILE_E2)
    print(f"    Event 1: {len(names_e1)} baris nama")
    print(f"    Event 2: {len(names_e2)} baris nama")

    # ── Step 2: Build mapping ─────────────────────────────────
    print("\n[STEP 2] Membangun tabel mapping nama → kode...")
    mapping_lower, mapping_display = build_mapping(names_e1, names_e2)
    print(f"    Total nama unik: {len(mapping_lower)}")
    print(f"    Contoh mapping:")
    for orig, code in list(mapping_display.items())[:5]:
        print(f"      '{orig}' → {code}")
    print(f"      ...")

    # ── Step 3: Save mapping CSV ──────────────────────────────
    print("\n[STEP 3] Menyimpan tabel mapping...")
    mapping_rows = [{"nama_asli": orig, "kode_anonim": code}
                    for orig, code in sorted(mapping_display.items())]
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(MAPPING_CSV, index=False, encoding="utf-8-sig")
    print(f"    Saved: {MAPPING_CSV} ({len(mapping_df)} baris)")

    # ── Step 4: Anonymize Event 1 ─────────────────────────────
    print("\n[STEP 4] Anonymisasi event1_raw.xlsx...")
    raw_e1 = anonymize_file(FILE_E1, mapping_lower)
    save_df_to_excel(raw_e1, FILE_E1)

    # ── Step 5: Anonymize Event 2 ─────────────────────────────
    print("\n[STEP 5] Anonymisasi event2_raw.xlsx...")
    raw_e2 = anonymize_file(FILE_E2, mapping_lower)
    save_df_to_excel(raw_e2, FILE_E2)

    # ── Step 6: Quick cross-event verification ────────────────
    print("\n[STEP 6] Verifikasi cross-event consistency...")
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

    # Check no real names remain
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
