# Teori Random Forest — Penjelasan Lengkap untuk Pemula
## Konteks: Prediksi Kehadiran Pelatihan Sushi & Onigiri, PT Cahaya Ladara Nusantara

---

## 1. Apa Itu Machine Learning?

Selama ini, ketika kita membuat program komputer untuk menyelesaikan suatu masalah, kita harus menuliskan aturan-aturannya secara eksplisit. Misalnya, jika seorang programmer ingin membuat program yang bisa menentukan apakah seorang peserta akan hadir atau tidak, ia mungkin akan menulis aturan seperti ini: "Jika jarak rumah lebih dari 15 km, maka prediksi Tidak Hadir. Jika jarak kurang dari 5 km, prediksi Hadir. Jika usia di atas 55 tahun, tambahkan risiko tidak hadir." Dan seterusnya.

Pendekatan seperti ini disebut pemrograman berbasis aturan. Masalahnya, untuk memprediksi kehadiran manusia yang perilakunya sangat kompleks dan tidak selalu mengikuti aturan sederhana, menulis aturan secara manual hampir mustahil. Bagaimana jika ada peserta yang tinggal 20 km jauhnya tetapi selalu hadir karena sangat termotivasi? Bagaimana jika ada peserta yang tinggal 1 km dari lokasi tetapi tidak pernah hadir karena kesibukan kerja? Setiap pengecualian berarti aturan baru, dan aturan baru mengundang pengecualian baru. Lingkaran ini tidak berujung.

Machine learning menawarkan pendekatan yang berbeda. Alih-alih kita yang menuliskan aturannya, kita memberikan data historis kepada komputer, yaitu data 300 peserta nyata dari dua event pelatihan Sushi dan Onigiri, lengkap dengan informasi siapa yang hadir dan siapa yang tidak, dan kita biarkan komputer menemukan sendiri pola-pola yang tersembunyi di dalam data tersebut. Komputer tidak diberitahu "jika jarak lebih dari sekian km maka tidak hadir." Komputer belajar sendiri bahwa jarak adalah faktor terpenting dengan memeriksa ribuan kombinasi angka dan pola di dalam data. Hasilnya adalah sebuah model matematika yang bisa membuat prediksi untuk peserta baru yang belum pernah dilihat sebelumnya.

---

## 2. Apa Itu Supervised Learning dan Klasifikasi Biner?

Machine learning memiliki banyak cabang. Yang digunakan dalam penelitian ini disebut Supervised Learning, atau Pembelajaran Terpandu. Disebut "terpandu" karena kita memberikan kepada komputer data yang sudah berlabel, artinya setiap baris data sudah diketahui jawabannya. Dari 300 baris data peserta PT Cahaya Ladara Nusantara, setiap baris sudah memiliki informasi apakah peserta tersebut "Hadir" atau "Tidak Hadir." Label inilah yang menjadi panduan bagi komputer untuk belajar.

Di dalam supervised learning, ada dua jenis tugas utama. Yang pertama adalah regresi, yaitu memprediksi sebuah angka. Yang kedua adalah klasifikasi, yaitu memprediksi sebuah kategori. Penelitian ini menggunakan klasifikasi karena hasilnya hanya ada dua kemungkinan: Hadir atau Tidak Hadir. Karena hanya ada dua kemungkinan, ini disebut klasifikasi biner, dari kata "bi" yang berarti dua.

Komputer belajar dari 240 data historis (80% dari total 300), lalu kemampuannya diuji menggunakan 60 data yang belum pernah dilihat sebelumnya (20% dari total 300). Proses belajar dari data lama untuk memprediksi kasus baru itulah inti dari seluruh sistem ini.

---

## 3. Apa Itu Pohon Keputusan (Decision Tree)?

Sebelum memahami Random Forest, kita perlu memahami bahan dasarnya terlebih dahulu: Pohon Keputusan atau Decision Tree.

Bayangkan seorang panitia yang berpengalaman mencoba menebak apakah seorang peserta akan hadir atau tidak. Ia tidak langsung memberikan jawaban. Ia mulai bertanya satu per satu. Pertanyaan pertamanya mungkin adalah: "Apakah jarak rumah peserta ini lebih dari 10 km?" Jika jawabannya Ya, kemungkinan besar peserta tidak hadir, tapi ia tidak berhenti di sini. Ia bertanya lagi: "Apakah usianya di bawah 35 tahun?" Jika Ya, mungkin peserta muda ini masih punya energi untuk menempuh perjalanan jauh. Jika tidak, risikonya semakin tinggi.

Itulah cara kerja Pohon Keputusan. Ia membuat serangkaian pertanyaan ya/tidak yang bercabang seperti pohon, hingga akhirnya sampai pada kesimpulan: Hadir atau Tidak Hadir.

Sebagai contoh konkret menggunakan data proyek ini: Di akar pohon (bagian paling atas), komputer mengajukan pertanyaan pertama tentang Jarak_km karena ini adalah faktor paling berpengaruh, yaitu sebesar 47,73% dalam penelitian ini. Pertanyaannya mungkin: "Apakah Jarak_km lebih dari 10?" Bagi peserta yang jawabannya Tidak (jarak di bawah 10 km), pohon bergerak ke cabang kiri. Di sana, pertanyaan berikutnya mungkin tentang Usia: "Apakah usia lebih dari 55 tahun?" Bagi peserta yang jawabannya Tidak (usia di bawah 55), pertanyaan berikutnya mungkin tentang riwayat hadir sebelumnya: "Apakah pernah hadir di event sebelumnya?" Setelah beberapa pertanyaan seperti ini, pohon akhirnya sampai pada ujungnya, yang disebut daun (leaf node). Daun inilah yang berisi prediksi akhir: apakah peserta ini diprediksi Hadir atau Tidak Hadir.

Semakin dalam sebuah pohon, semakin spesifik pertanyaannya, dan semakin akurat prediksinya untuk data yang sudah dikenal. Namun, seperti yang akan kita lihat sebentar lagi, pohon yang terlalu dalam justru bisa menjadi masalah.

---

## 4. Gini Impurity — Bagaimana Pohon Memilih Pertanyaan Terbaik?

Ketika sebuah Pohon Keputusan harus memilih pertanyaan pertama yang paling baik, ia membutuhkan cara untuk mengukur seberapa baik sebuah pertanyaan memisahkan data. Ukuran ini disebut Gini Impurity atau "tingkat ketidakmurnian Gini."

Cara termudah untuk memahaminya adalah dengan menggunakan analogi keranjang buah. Bayangkan kita memiliki dua keranjang. Keranjang pertama berisi 100% apel merah, tidak ada satupun buah lain. Keranjang ini "murni" karena isinya seragam. Dalam bahasa Gini, keranjang ini memiliki nilai ketidakmurnian 0, artinya sangat murni, tidak ada campuran. Keranjang kedua berisi 50% apel merah dan 50% apel hijau, tercampur sempurna. Ini adalah keranjang paling "tidak murni" yang bisa ada, dengan nilai ketidakmurnian mendekati 0.5.

Sekarang bayangkan kita memiliki 300 data peserta PT Cahaya Ladara Nusantara sebelum dipecah oleh pertanyaan apapun. Ada 236 yang Hadir dan 64 yang Tidak Hadir, tercampur menjadi satu. Ini seperti keranjang yang sangat tidak murni. Tugas Pohon Keputusan adalah menemukan pertanyaan yang paling efektif memisahkan kedua kelompok ini menjadi dua sub-kelompok yang masing-masing lebih murni dari sebelumnya.

Ketika komputer mencoba pertanyaan "Apakah Jarak_km lebih dari 10?", hasilnya adalah: satu kelompok (jarak dekat) sebagian besar berisi peserta yang Hadir, dan satu kelompok lagi (jarak jauh) memiliki proporsi Tidak Hadir yang lebih tinggi. Kedua kelompok yang dihasilkan lebih murni dari campuran aslinya. Ini adalah pertanyaan yang baik. Ketika komputer mencoba pertanyaan "Apakah Event_ID sama dengan 1?", mungkin hasilnya tidak terlalu berbeda antara kedua kelompok. Ini adalah pertanyaan yang kurang baik.

Komputer memilih pertanyaan yang paling dramatis mengurangi ketidakmurnian. Pertanyaan tentang Jarak_km terpilih sebagai pertanyaan pertama bukan karena ada yang memberitahu komputer demikian, melainkan karena secara matematis, Jarak_km adalah faktor yang paling efektif memisahkan peserta yang Hadir dari yang Tidak Hadir dalam data 300 orang tersebut.

---

## 5. Mengapa Satu Pohon Keputusan Tidak Cukup?

Jika Pohon Keputusan begitu logis dan sistematis, mengapa tidak cukup hanya menggunakan satu pohon saja?

Masalahnya ada pada fenomena yang disebut overfitting, atau "terlalu menyesuaikan diri" dengan data latih. Bayangkan seorang mahasiswa yang belajar untuk ujian dengan cara menghafal jawaban dari soal-soal latihan tahun lalu, tanpa benar-benar memahami materinya. Ketika ujian tiba dengan soal-soal baru, mahasiswa ini akan kesulitan karena ia tidak punya pemahaman mendalam, hanya hafalan.

Hal yang sama bisa terjadi pada Pohon Keputusan. Jika pohon dibiarkan tumbuh terlalu dalam, ia akan menghafal setiap detail dari 240 data pelatihan, termasuk detail-detail yang sebenarnya hanya kebetulan, bukan pola nyata. Misalnya, mungkin kebetulan dalam data latih ada tiga peserta berusia tepat 43 tahun yang semuanya tidak hadir, dan pohon ini belajar bahwa "usia 43 tahun selalu tidak hadir." Tentu saja ini adalah aturan yang salah secara umum, tapi pohon menemukannya karena terlalu detail memperhatikan data latih.

Ketika pohon seperti ini dihadapkan pada 60 data uji yang belum pernah dilihat, performanya bisa jauh di bawah harapan karena aturan-aturan yang dihafalnya tidak berlaku untuk kasus baru. Inilah kelemahan fundamental dari satu pohon keputusan tunggal.

---

## 6. Bootstrap Sampling — Setiap Pohon Belajar dari Data yang Berbeda

Random Forest mengatasi masalah overfitting dengan cara yang cerdas. Alih-alih melatih satu pohon besar menggunakan semua 300 data, Random Forest melatih banyak pohon, dalam penelitian ini sebanyak 100 pohon, dan setiap pohon dilatih menggunakan sampel data yang berbeda.

Caranya disebut bootstrap sampling. Dari 300 data peserta, sistem secara acak mengambil sekitar 200-240 baris untuk melatih satu pohon. Proses pengambilan ini dilakukan "dengan pengembalian", artinya sebuah baris data bisa terpilih lebih dari sekali untuk pohon yang sama. Kemudian, untuk pohon kedua, sistem kembali mengambil sampel acak yang berbeda dari 300 data tersebut. Lalu pohon ketiga dengan sampel berbeda lagi, dan seterusnya hingga 100 pohon.

Hasilnya adalah 100 pohon yang masing-masing memiliki sedikit "pengalaman hidup" yang berbeda. Pohon pertama mungkin banyak belajar dari peserta-peserta yang tinggal jauh di Jakarta Utara. Pohon ke-57 mungkin banyak belajar dari peserta berusia muda dari Jakarta Timur. Karena setiap pohon memiliki sudut pandang yang sedikit berbeda, keputusan kolektif mereka jauh lebih bijaksana daripada keputusan satu pohon tunggal.

---

## 7. Random Feature Selection — Setiap Pertanyaan Dari Pilihan yang Berbeda

Selain baris data yang berbeda, Random Forest menambahkan satu lapisan keacakan lagi. Ketika setiap pohon harus memilih pertanyaan terbaik di setiap cabang, ia tidak diizinkan untuk mempertimbangkan keenam fitur sekaligus. Sebagai gantinya, ia hanya boleh mempertimbangkan sebagian kecil dari keenam fitur tersebut, misalnya hanya tiga fitur yang dipilih secara acak.

Mengapa ini penting? Tanpa pembatasan ini, hampir setiap pohon akan mengajukan pertanyaan tentang Jarak_km di akar pohonnya, karena Jarak_km memang faktor paling berpengaruh. Hasilnya, semua 100 pohon akan sangat mirip satu sama lain. Pohon-pohon yang mirip memberikan suara yang sangat mirip, sehingga tidak ada keuntungan dari memiliki banyak pohon.

Dengan pemilihan fitur secara acak, kadang-kadang Jarak_km tidak termasuk dalam pilihan yang tersedia untuk sebuah cabang, sehingga pohon itu terpaksa belajar membuat keputusan berdasarkan Usia, Status Pendaftaran, atau Riwayat Hadir Sebelumnya. Ini memaksa setiap pohon untuk menjadi lebih beragam dan mandiri. Keberagaman inilah yang membuat kolektif 100 pohon jauh lebih kuat dari satu pohon terbaik sekalipun.

---

## 8. Hutan Lengkap — Bagaimana 100 Pohon Memberikan Satu Jawaban?

Setelah 100 pohon selesai dilatih, proses prediksi untuk seorang peserta baru sangat sederhana: keenam angka dari data peserta tersebut dikirimkan secara bersamaan ke semua 100 pohon. Setiap pohon mengikuti percabangan pertanyaannya sendiri berdasarkan data yang diterima, dan setiap pohon memberikan satu suara: Hadir atau Tidak Hadir.

Setelah semua 100 suara terkumpul, sistem menghitung. Misalnya, 67 pohon mengatakan Hadir dan 33 pohon mengatakan Tidak Hadir. Prediksi akhir adalah Hadir karena mayoritas suara mengarah ke sana.

Di sinilah angka persentase yang muncul di aplikasi berasal. Persentase 67% yang ditampilkan sebagai "Peluang Hadir" bukan angka ajaib. Ini adalah hasil pembagian sederhana: 67 pohon yang memilih Hadir dibagi dengan total 100 pohon, sama dengan 0.67, atau 67%. Dalam bahasa teknis, angka ini disebut predict_proba, yaitu probabilitas prediksi. Jika hanya 20 pohon yang mengatakan Hadir dan 80 mengatakan Tidak Hadir, maka peluang Hadir ditampilkan sebagai 20% dan peluang Tidak Hadir sebagai 80%.

Sistem ini bekerja seperti dewan juri yang terdiri dari 100 hakim berpengalaman dengan sudut pandang berbeda-beda, bukan hakim tunggal yang mungkin memiliki bias.

---

## 9. Feature Importance — Fitur Mana yang Paling Sering Ditanya?

Setelah 100 pohon selesai dilatih, sistem bisa menghitung seberapa sering setiap fitur digunakan di dekat akar pohon, yaitu di posisi pertanyaan-pertanyaan awal yang paling menentukan. Fitur yang sering muncul di dekat akar pohon dan berhasil mengurangi ketidakmurnian paling banyak mendapatkan nilai importance yang tinggi.

Dalam penelitian ini, hasilnya adalah: Jarak_km mendapatkan importance 47,73%, artinya pertanyaan tentang jarak muncul di posisi paling strategis di hampir setengah dari semua 100 pohon. Usia mendapatkan importance 36%, artinya pertanyaan tentang usia adalah pertanyaan kedua yang paling sering muncul di posisi penting.

Mengapa ini masuk akal? Jarak adalah hambatan fisik yang paling nyata. Bagi anggota PT Cahaya Ladara Nusantara yang harus hadir di Halim Perdanakusuma, seorang yang tinggal 1 km jauhnya memiliki hambatan yang sangat berbeda dibanding seseorang yang tinggal 25 km jauhnya di tengah kemacetan Jakarta. Perbedaan hambatan ini sangat konsisten di antara 300 peserta dalam data, sehingga pertanyaan tentang jarak berulang kali terbukti sebagai yang paling efektif memisahkan dua kelompok.

Usia, di posisi kedua, juga sangat masuk akal. Kemampuan fisik, komitmen terhadap jadwal, dan motivasi untuk belajar keterampilan baru sering kali berkaitan dengan tahap kehidupan seseorang. Data dari 300 peserta ini membuktikan bahwa usia menyumbang sinyal yang kuat dan konsisten dalam membedakan yang hadir dari yang tidak.

---

## 10. class_weight Balanced — Memberi Perhatian Setara pada Minoritas

Dalam data 300 peserta PT Cahaya Ladara Nusantara, sebanyak 236 peserta hadir (78,67%) dan hanya 64 yang tidak hadir (21,33%). Ini adalah ketidakseimbangan yang cukup signifikan.

Jika model dibiarkan belajar tanpa koreksi apapun, ia akan secara alami condong untuk selalu memprediksi Hadir. Mengapa? Karena dari perspektif komputer, jika sebuah pohon selalu menjawab "Hadir" untuk semua orang, ia akan benar 78,67% dari waktu, yang terdengar seperti angka yang bagus. Model tidak peduli bahwa ia sama sekali tidak berguna untuk mengidentifikasi peserta yang akan absen. Ia hanya peduli untuk meminimalkan jumlah kesalahan secara keseluruhan.

Analogi yang tepat adalah seorang guru yang memiliki 30 murid yang lulus dan 5 murid yang tidak lulus. Jika guru ini hanya memperhatikan 30 murid yang lulus, nilai rata-rata kelasnya akan terlihat sangat bagus. Tapi 5 murid yang paling membutuhkan bantuan justru terabaikan. Secara keseluruhan, kelasnya terlihat baik di atas kertas, tapi ada yang gagal lolos begitu saja.

Parameter class_weight balanced dalam penelitian ini berfungsi seperti instruksi kepada guru tersebut: "Perlakukan setiap murid yang tidak lulus sebagai sama pentingnya dengan setiap murid yang lulus, terlepas dari berapa banyak jumlahnya." Secara teknis, ini berarti kesalahan memprediksi peserta yang Tidak Hadir diberi penalti yang lebih besar selama pelatihan, sehingga model dipaksa untuk lebih memperhatikan 64 peserta minoritas yang tidak hadir, bukan hanya 236 peserta mayoritas yang hadir.

Tanpa pengaturan ini, sistem mungkin akan mengabaikan hampir semua peserta yang tidak hadir. Dengan pengaturan ini, setidaknya model mencoba lebih keras untuk mendeteksi mereka, meskipun dengan data yang hanya 300 baris, hasilnya masih belum sempurna.

---

## 11. Bagaimana Prediksi Akhir Bekerja di Aplikasi Ini?

Ketika seorang pengguna membuka tab "Prediksi Kehadiran" di aplikasi web PT Cahaya Ladara Nusantara dan mengisi formulir, inilah yang terjadi di balik layar secara berurutan:

Pertama, pengguna memasukkan keenam informasi peserta: jenis kelamin, usia dalam tahun, jarak dari rumah ke lokasi event dalam kilometer, status pendaftaran (terdaftar atau tidak), nomor event (1 untuk 9 November atau 2 untuk 16 November), dan apakah peserta pernah hadir di event sebelumnya. Keenam informasi ini diubah menjadi angka. Jenis kelamin "Perempuan" menjadi angka 0, "Laki-Laki" menjadi 1. "Terdaftar" menjadi 1, "Tidak Terdaftar" menjadi 0. Riwayat hadir menjadi 1, tidak hadir menjadi 0. Hasilnya adalah sebuah barisan enam angka, misalnya seperti [0, 35, 3.5, 1, 2, 1] yang mewakili: perempuan, usia 35 tahun, jarak 3.5 km, terdaftar, Event 2, pernah hadir sebelumnya.

Keenam angka ini kemudian dikirimkan secara bersamaan ke seluruh 100 pohon yang sudah dilatih dan tersimpan di dalam file model (random_forest_model.pkl berukuran 951 KB di dalam folder models). Setiap pohon menelusuri percabangan pertanyaannya sendiri berdasarkan keenam angka tersebut dan memberikan satu suara.

Setelah semua 100 suara terkumpul, sistem menghitung suara Hadir dan Tidak Hadir. Jika 82 pohon mengatakan Hadir dan 18 pohon mengatakan Tidak Hadir, maka aplikasi menampilkan "Diprediksi Hadir" dengan "Peluang Hadir: 82.0%" dan "Peluang Tidak Hadir: 18.0%." Angka-angka inilah yang terlihat oleh pengguna aplikasi, disertai grafik batang yang memperlihatkan perbandingan kedua peluang tersebut secara visual.

Seluruh proses dari penekanan tombol "Prediksi Sekarang" hingga hasil muncul di layar berlangsung hanya dalam sepersekian detik, karena komputer modern bisa menjalankan 100 pohon secara hampir bersamaan. Inilah keunggulan sistem otomatis dibanding seorang panitia yang harus mempertimbangkan setiap faktor secara manual satu per satu.

---

*Seluruh penjelasan dalam dokumen ini didasarkan pada implementasi nyata dalam repositori skripsi ini. Angka-angka seperti 100 pohon, 300 baris data, 47.73% untuk Jarak_km, dan 36.00% untuk Usia seluruhnya berasal dari file konfigurasi dan hasil evaluasi yang tersimpan di dalam repositori.*
