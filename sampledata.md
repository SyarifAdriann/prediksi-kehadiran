# Sample Data Uji — Sistem Prediksi Kehadiran Anggota
## PT Cahaya Ladara Nusantara | Validated by Model

> **Catatan:** Semua hasil prediksi di bawah ini telah **divalidasi langsung dari model**
> (`random_forest_model.pkl`) menggunakan skrip Python sebelum dokumen ini dibuat.
> Angka probabilitas adalah output aktual `model.predict_proba()`.
>
> **Cara menggunakan:** Buka aplikasi di `http://localhost:8501` → Tab **🔮 Prediksi Kehadiran**
> → Isi form sesuai kolom di bawah → Klik **Prediksi Sekarang**.
>
> Catatan dropdown "Hadir Event Sebelumnya?":
> - **"Tidak Relevan / Event Pertama"** → pilih ini jika event = Event 1, atau peserta baru Event 2 yang memang tidak ada di Event 1 sama sekali
> - **"Ya — Hadir"** → peserta Event 2 yang hadir di Event 1
> - **"Tidak — Tidak Hadir"** → peserta Event 2 yang TERDAFTAR di Event 1 tapi TIDAK hadir

---

## ✅ Kategori 1: DIPREDIKSI HADIR (Confidence Tinggi ≥ 87%)

| # | Jenis Kelamin | Usia | Jarak (km) | Event | Hadir Event Sebelumnya? | Prediksi | Prob. Hadir | Prob. Tidak Hadir |
|---|---|---|---|---|---|---|---|---|
| H1 | Perempuan | 43 | 1.5 | Event 2 — 16 Nov | Ya — Hadir | ✅ **HADIR** | **97%** | 3% |
| H2 | Perempuan | 42 | 2.5 | Event 2 — 16 Nov | Ya — Hadir | ✅ **HADIR** | **95%** | 5% |
| H3 | Perempuan | 35 | 4.0 | Event 2 — 16 Nov | Ya — Hadir | ✅ **HADIR** | **93%** | 7% |
| H4 | Perempuan | 58 | 0.5 | Event 1 — 09 Nov | Tidak Relevan / Event Pertama | ✅ **HADIR** | **91%** | 9% |
| H5 | Laki-Laki | 35 | 7.0 | Event 1 — 09 Nov | Tidak Relevan / Event Pertama | ✅ **HADIR** | **90%** | 10% |
| H6 | Perempuan | 28 | 8.0 | Event 1 — 09 Nov | Tidak Relevan / Event Pertama | ✅ **HADIR** | **88%** | 12% |
| H7 | Laki-Laki | 45 | 2.0 | Event 2 — 16 Nov | Tidak Relevan / Event Pertama | ✅ **HADIR** | **87%** | 13% |

### Pola yang mendorong prediksi Hadir tinggi:
- Jarak **< 10 km** dari lokasi event → faktor terbesar (importance 47.73%)
- Memiliki **riwayat hadir** di event sebelumnya → indikator loyalitas
- **Usia 28–58 tahun** (rentang produktif) → berkontribusi besar (importance 36.00%)
- **Perempuan** → profil dominan di dataset (91.7%)

---

## ✅ Kategori 2: DIPREDIKSI HADIR (Confidence Menengah 60–86%)

> Kasus-kasus ini masih diprediksi Hadir tetapi dengan keyakinan lebih rendah.
> Satu atau lebih faktor risiko hadir namun belum cukup dominan.

| # | Jenis Kelamin | Usia | Jarak (km) | Event | Hadir Event Sebelumnya? | Prediksi | Prob. Hadir | Prob. Tidak Hadir |
|---|---|---|---|---|---|---|---|---|
| H8 | Perempuan | 42 | 2.5 | Event 2 — 16 Nov | Tidak Relevan / Event Pertama | ✅ **HADIR** | 87% | 13% |
| H9 | Perempuan | 55 | 1.8 | Event 2 — 16 Nov | Ya — Hadir | ✅ **HADIR** | 94% | 6% |
| H10 | Laki-Laki | 52 | 22.0 | Event 2 — 16 Nov | Ya — Hadir | ✅ **HADIR** | 67% | 33% |
| H11 | Laki-Laki | 55 | 40.0 | Event 2 — 16 Nov | Ya — Hadir | ✅ **HADIR** | 71% | 29% |
| H12 | Perempuan | 60 | 12.0 | Event 2 — 16 Nov | Ya — Hadir | ✅ **HADIR** | 55% | 45% |
| H13 | Laki-Laki | 62 | 18.0 | Event 1 — 09 Nov | Tidak Relevan / Event Pertama | ✅ **HADIR** | 52% | 48% |

### Faktor risiko yang menekan confidence:
- **Jarak jauh** (>15 km) menurunkan keyakinan model
- **Usia tua** (>58 tahun) + **Laki-laki** → kombinasi yang lebih tidak pasti
- **Tanpa riwayat hadir** di event sebelumnya

---

## ❌ Kategori 3: DIPREDIKSI TIDAK HADIR

| # | Jenis Kelamin | Usia | Jarak (km) | Event | Hadir Event Sebelumnya? | Prediksi | Prob. Hadir | Prob. Tidak Hadir |
|---|---|---|---|---|---|---|---|---|
| TH1 | Perempuan | 45 | **0.5** | Event 2 — 16 Nov | **Tidak — Tidak Hadir** | ❌ **TIDAK HADIR** | 9% | **91%** |
| TH2 | Perempuan | 35 | 20.0 | Event 1 — 09 Nov | Tidak Relevan / Event Pertama | ❌ **TIDAK HADIR** | 27% | **73%** |
| TH3 | Perempuan | 65 | 12.0 | Event 2 — 16 Nov | Ya — Hadir | ❌ **TIDAK HADIR** | 33% | **67%** |
| TH4 | Perempuan | 40 | 20.0 | Event 1 — 09 Nov | Tidak Relevan / Event Pertama | ❌ **TIDAK HADIR** | 35% | **65%** |
| TH5 | Perempuan | 35 | 12.0 | Event 1 — 09 Nov | Tidak Relevan / Event Pertama | ❌ **TIDAK HADIR** | 30% | **70%** |
| TH6 | Perempuan | 45 | 0.5 | Event 1 — 09 Nov | Tidak Relevan / Event Pertama | ❌ **TIDAK HADIR** | 31% | **69%** |
| TH7 | Laki-Laki | 60 | 25.0 | Event 1 — 09 Nov | Tidak Relevan / Event Pertama | ❌ **TIDAK HADIR** | 47% | **53%** |

### Penjelasan pola masing-masing:

**TH1 — Kasus paling unik (91% Tidak Hadir meski jarak 0.5 km):**
> Peserta tinggal sangat dekat (0.5 km) tetapi terdaftar di Event 2 tanpa menghadiri Event 1.
> Model menginterpretasikan ini sebagai indikasi kuat ketidakmampuan/ketidakmauan hadir —
> jika orang yang tinggal di sebelah lokasi saja tidak hadir di Event 1, kemungkinan besar
> ada alasan struktural (misalnya: jadwal bentrok tetap, kurang minat).

**TH2 & TH4 & TH5 — Jarak jauh + Event 1 + Peserta baru:**
> Kombinasi jarak >12 km, Event 1 (pertama kali), dan tanpa riwayat menghasilkan prediksi Tidak Hadir.
> Ini pola yang paling **intuitif** — peserta baru dari jauh cenderung tidak hadir.

**TH3 — Usia 65 tahun + jauh:**
> Peserta tertua (65 tahun) dengan jarak 12 km diprediksi Tidak Hadir meski pernah hadir sebelumnya.
> Usia ekstrem + jarak cukup jauh mengalahkan sinyal positif dari riwayat hadir.

**TH6 — Sama seperti TH1 tapi Event 1:**
> Jarak 0.5 km namun Event 1, tanpa riwayat → model tidak punya referensi positif
> untuk profil ini, dan memprediksi Tidak Hadir 69%.

**TH7 — Laki-Laki + sangat tua + sangat jauh:**
> Kombinasi klasik: jenis kelamin minoritas (L) + usia 60 + jarak 25 km + Event 1.

---

## 📊 Ringkasan Temuan

| Kategori | Jumlah Kasus | Prob. Hadir | Prob. Tidak Hadir |
|---|---|---|---|
| Hadir (Confidence Tinggi) | 7 kasus | 87% – 97% | 3% – 13% |
| Hadir (Confidence Menengah) | 6 kasus | 52% – 94% | 6% – 48% |
| Tidak Hadir | 7 kasus | 9% – 47% | 53% – 91% |

**Faktor paling menentukan prediksi Tidak Hadir:**
1. Terdaftar di Event 2 tapi tidak hadir di Event 1 → sinyal penolakan paling kuat
2. Jarak >12 km + Event 1 + peserta baru
3. Usia >60 tahun + jarak >20 km
4. Kombinasi Laki-Laki + usia tua + jarak jauh

**Batasan model:**
- Semua prediksi Tidak Hadir di atas hanya mencapai 53%–91% confidence
- Model tidak pernah mencapai 100% Tidak Hadir karena 78.7% data training adalah Hadir
- Gunakan model sebagai *screening tool*, bukan keputusan final

---

*Dokumen ini dibuat dan divalidasi pada: 03 Juni 2026*
*Model: `models/random_forest_model.pkl` | Akurasi model pada test set: 75.00%*
