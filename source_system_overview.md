# Penjelasan Lengkap Sistem Prediksi Kehadiran Anggota
## PT Cahaya Ladara Nusantara — Untuk Pembaca Umum

---

## 1. Tentang PT Cahaya Ladara Nusantara dan Masalah yang Dihadapi

PT Cahaya Ladara Nusantara adalah sebuah perusahaan yang secara rutin mengadakan pelatihan untuk para anggotanya. Pelatihan yang menjadi fokus penelitian ini adalah pelatihan memasak makanan viral, yaitu Sushi dan Onigiri, yang diadakan pada bulan November 2025.

Seperti banyak organisasi lain yang mengadakan kegiatan, PT Cahaya Ladara Nusantara menghadapi satu masalah yang terdengar sederhana tapi dampaknya cukup besar: mereka tidak bisa tahu sebelumnya siapa saja yang benar-benar akan hadir di hari pelaksanaan. Sudah mendaftar bukan berarti pasti datang. Banyak peserta yang mendaftar tetapi akhirnya tidak hadir karena berbagai alasan, seperti jarak yang jauh dari rumah, kondisi tubuh yang tidak memungkinkan, atau kesibukan mendadak.

Akibatnya, panitia sering kali menyiapkan bahan makanan, tempat duduk, dan tenaga instruktur berdasarkan jumlah pendaftar, bukan jumlah yang benar-benar hadir. Ini menyebabkan pemborosan sumber daya yang bisa dihindari apabila ada alat bantu untuk memprediksi kehadiran lebih awal.

Itulah latar belakang dari skripsi ini. Tujuannya adalah membuat sebuah sistem komputer yang bisa mempelajari pola data peserta dan memprediksi: apakah peserta ini kemungkinan besar akan hadir atau tidak hadir?

---

## 2. Apa yang Dilakukan Sistem Ini?

Sistem ini adalah sebuah aplikasi berbasis web yang bisa diakses melalui browser. Pengguna cukup memasukkan data seorang peserta, misalnya jenis kelaminnya, usianya, jarak rumahnya ke lokasi event, dan beberapa informasi lain, lalu sistem akan menjawab dua hal: pertama, apakah peserta tersebut diprediksi Hadir atau Tidak Hadir, dan kedua, seberapa yakin model dengan jawabannya itu, ditampilkan dalam bentuk persentase.

Contohnya, jika seorang peserta berusia 40 tahun, tinggal 2 kilometer dari lokasi pelatihan, sudah terdaftar secara resmi, dan pernah hadir di event sebelumnya, maka sistem mungkin akan menjawab "Diprediksi Hadir" dengan tingkat keyakinan 88%. Angka ini bukan tebakan sembarangan. Angka ini dihasilkan oleh sebuah model matematika yang sudah belajar dari data 300 peserta nyata sebelumnya.

---

## 3. Data yang Digunakan

Untuk melatih sistem ini, peneliti menggunakan data dari dua event pelatihan yang sesungguhnya, yaitu Event 1 pada tanggal 9 November 2025 dan Event 2 pada tanggal 16 November 2025. Keduanya adalah pelatihan Sushi dan Onigiri yang diselenggarakan oleh PT Cahaya Ladara Nusantara di lokasi Halim Perdanakusuma, Jakarta Timur.

Dari kedua event tersebut, total data yang dikumpulkan adalah 300 baris, masing-masing mewakili satu orang peserta. Setiap baris berisi informasi tentang peserta tersebut, mulai dari nama, alamat, hingga apakah mereka akhirnya hadir atau tidak.

Dari semua informasi yang ada, sistem hanya menggunakan 6 faktor utama untuk membuat prediksi. Keenam faktor ini disebut "fitur" dalam bahasa ilmu data. Berikut penjelasannya:

**Jenis Kelamin:** Apakah peserta laki-laki atau perempuan. Dalam data ini, 91,7% peserta adalah perempuan, sehingga pengaruh fitur ini terhadap prediksi cukup kecil karena tidak banyak variasi.

**Usia:** Berapa tahun usia peserta saat mendaftar. Ini adalah faktor yang ternyata sangat berpengaruh, seperti yang akan dijelaskan lebih lanjut di bagian bawah.

**Jarak dari Rumah ke Lokasi Event (dalam kilometer):** Seberapa jauh peserta harus menempuh perjalanan untuk sampai ke lokasi pelatihan. Ini adalah faktor paling berpengaruh dalam seluruh sistem.

**Status Pendaftaran:** Apakah peserta sudah terdaftar secara resmi atau belum. Peserta yang terdaftar resmi cenderung lebih berkomitmen untuk datang.

**Event ID:** Apakah peserta ini mengikuti Event 1 atau Event 2. Kedua event memiliki karakteristik yang sedikit berbeda, misalnya tingkat promosi atau kondisi cuaca pada hari itu.

**Hadir di Event Sebelumnya:** Apakah peserta ini pernah hadir di event PT Cahaya Ladara Nusantara sebelumnya? Peserta yang pernah hadir sebelumnya menunjukkan loyalitas dan cenderung datang kembali.

---

## 4. Apa Itu Random Forest?

Metode yang digunakan dalam sistem ini adalah Random Forest, atau dalam bahasa Indonesia dapat disebut sebagai "Hutan Acak". Berikut penjelasannya dalam satu paragraf sederhana:

Bayangkan Anda ingin memutuskan apakah sebaiknya membawa payung saat keluar rumah. Anda tidak hanya bertanya pada satu orang, melainkan bertanya pada 100 orang sekaligus. Setiap orang melihat situasi dari sudut pandangnya sendiri, misalnya ada yang fokus pada warna langit, ada yang memperhatikan arah angin, dan ada yang melihat prakiraan cuaca. Setelah semua orang memberikan pendapatnya, Anda mengikuti suara terbanyak. Itulah yang dilakukan Random Forest. Sistem ini tidak menggunakan satu "pohon keputusan" saja, melainkan 100 pohon keputusan yang masing-masing melihat data dari sudut pandang yang sedikit berbeda. Hasil akhirnya adalah keputusan berdasarkan suara mayoritas dari 100 pohon tersebut, sehingga hasilnya jauh lebih akurat dan tidak mudah salah karena satu sudut pandang yang keliru.

---

## 5. Bagaimana Data Dibersihkan Sebelum Digunakan?

Data mentah yang diperoleh dari formulir pendaftaran peserta tidak langsung bisa digunakan oleh komputer. Ada banyak masalah kecil yang harus diselesaikan terlebih dahulu. Proses pembersihan ini adalah salah satu bagian terpenting dari penelitian.

**Masalah Pertama: Format Jarak yang Tidak Konsisten.** Para peserta mengisi kolom jarak dengan format yang berbeda-beda. Ada yang menulis "12 KM", ada yang menulis "400 M" (meter, bukan kilometer), ada yang menggunakan koma sebagai desimal seperti "13,8 KM", dan ada yang hanya menulis angka saja. Sistem ini secara otomatis membaca semua format tersebut dan mengubahnya menjadi satu format standar dalam satuan kilometer. Jadi "400 M" diubah menjadi "0.4 km", dan "13,8 KM" diubah menjadi "13.8 km".

**Masalah Kedua: Jarak yang Tidak Masuk Akal.** Beberapa peserta mengisi jarak yang jauh di luar nalar, misalnya seseorang yang tinggal di Kecamatan Makasar (yang secara geografis sangat dekat dengan Halim) mengisi jarak 80 km. Ini jelas merupakan kesalahan pengisian. Untuk menangani ini, sistem menghitung sendiri perkiraan jarak berdasarkan koordinat geografis kecamatan tempat tinggal peserta menggunakan rumus matematika yang disebut Haversine, yaitu rumus yang menghitung jarak dua titik di permukaan bumi secara akurat. Jika jarak yang diisi peserta lebih dari tiga kali lipat perkiraan wajar dan lebih dari 10 km di atas perkiraan tersebut, nilai yang diisi peserta diganti dengan nilai perkiraan yang lebih masuk akal.

**Masalah Ketiga: Data Kosong.** Beberapa peserta tidak mengisi semua kolom. Untuk kolom jarak yang kosong, sistem menggunakan nilai median, yaitu nilai tengah dari semua data jarak yang ada. Median digunakan karena lebih tahan terhadap nilai-nilai ekstrem dibanding rata-rata.

**Masalah Keempat: Mengetahui Riwayat Kehadiran.** Fitur "hadir di event sebelumnya" tidak ada secara langsung di data mentah. Untuk membuat fitur ini, sistem mencocokkan nama-nama peserta Event 2 dengan daftar peserta yang hadir di Event 1. Jika nama peserta ditemukan di Event 1 dan statusnya adalah "Hadir", maka fitur ini diisi dengan nilai 1 (ya, pernah hadir). Jika tidak ditemukan atau statusnya Tidak Hadir, fitur ini diisi dengan nilai 0.

---

## 6. Apa Artinya Hasil 75% Akurasi?

Setelah model selesai dilatih menggunakan 240 data (80% dari total 300), sistem kemudian diuji menggunakan 60 data yang belum pernah dilihat model sebelumnya (20% dari total 300). Hasilnya adalah akurasi sebesar 75%, artinya dari 60 data uji tersebut, model berhasil memprediksi dengan benar sebanyak 45 orang, dan salah pada 15 orang.

Sekarang, apakah 75% itu bagus atau buruk?

Ini adalah pertanyaan penting yang harus dijawab dengan jujur. Jika seseorang tanpa menggunakan sistem apapun hanya menebak bahwa semua peserta akan hadir, maka tingkat keakuratannya adalah 78,67%, karena memang sebanyak 78,67% peserta dalam data ini benar-benar hadir. Artinya, tebakan buta "semua hadir" sebenarnya menghasilkan angka yang sedikit lebih tinggi dari model ini.

Namun, ada hal yang perlu dipahami. Tebakan buta tersebut sama sekali tidak berguna secara praktis. Siapapun yang selalu menebak "hadir" tidak akan pernah bisa mengidentifikasi satu pun peserta yang akan absen. Sementara itu, model Random Forest dalam sistem ini, meskipun akurasinya sedikit lebih rendah secara keseluruhan, setidaknya berhasil mengidentifikasi 2 orang dari 13 yang benar-benar tidak hadir. Ini adalah kemampuan yang tidak dimiliki oleh tebakan buta manapun. Jadi model ini memiliki nilai praktis, terutama jika digunakan sebagai alat bantu awal, bukan sebagai keputusan final. Semakin banyak data yang dikumpulkan dari event-event berikutnya, semakin akurat model ini akan menjadi.

---

## 7. Apa yang Bisa Dilakukan di Aplikasi Web Ini?

Aplikasi web yang dibuat dalam penelitian ini dapat diakses melalui browser dan memiliki empat halaman utama yang disebut "tab":

**Tab Prediksi Kehadiran** adalah inti dari seluruh sistem. Di sini, pengguna memasukkan data satu peserta: jenis kelamin, usia, jarak dari rumah ke lokasi event, event yang diikuti, dan apakah peserta pernah hadir sebelumnya. Setelah mengklik tombol "Prediksi Sekarang", sistem akan menampilkan jawaban apakah peserta tersebut diprediksi Hadir atau Tidak Hadir, beserta persentase keyakinannya untuk kedua kemungkinan tersebut.

**Tab Dashboard** menampilkan ringkasan data seluruh peserta dalam bentuk grafik dan angka. Pengguna bisa melihat berapa total peserta, berapa yang hadir, berapa yang tidak hadir, bagaimana distribusi usia peserta, dan bagaimana sebaran jarak rumah peserta ke lokasi event. Semua informasi ini disajikan dalam bentuk visual yang mudah dipahami.

**Tab Evaluasi Model** menampilkan laporan kinerja model secara teknis. Di sini tersedia akurasi, presisi, recall, F1-score, tabel perbandingan per kelas, serta gambar Confusion Matrix dan Feature Importance. Pengguna juga dapat mengunduh laporan evaluasi ini dalam format PDF.

**Tab Log Prediksi** menyimpan riwayat semua prediksi yang pernah dilakukan selama menggunakan aplikasi. Setiap prediksi dicatat beserta waktu, data peserta, dan hasilnya. Riwayat ini bisa diunduh sebagai file CSV untuk keperluan dokumentasi.

---

## 8. Mengapa Jarak dan Usia Begitu Dominan?

Dari keenam faktor yang digunakan, ternyata dua faktor mendominasi lebih dari 83% dari seluruh keputusan model: Jarak sejauh 47,73% dan Usia sebesar 36%.

Dominasi Jarak tidak mengejutkan. Semakin jauh seseorang tinggal dari lokasi event, semakin besar usaha fisik, waktu, dan biaya transportasi yang harus dikeluarkan. Seorang peserta yang tinggal 1 km dari lokasi cukup berjalan kaki. Seorang peserta yang tinggal 25 km mungkin harus menyetir hampir satu jam dalam kemacetan Jakarta. Secara naluriah, semakin besar hambatan fisik, semakin besar kemungkinan seseorang urung hadir.

Dominasi Usia juga masuk akal. Peserta dengan usia produktif, yaitu sekitar 30 hingga 50 tahun, cenderung lebih konsisten dalam kehadiran, mungkin karena mereka lebih terorganisir dalam mengelola jadwal, atau karena mereka merasakan manfaat pelatihan secara langsung untuk kehidupan mereka. Peserta yang jauh lebih muda atau jauh lebih tua mungkin menghadapi tantangan berbeda yang memengaruhi kemampuan mereka untuk hadir.

Sementara itu, fitur seperti Riwayat Hadir Sebelumnya dan Status Pendaftaran memberikan kontribusi yang lebih kecil, masing-masing sekitar 3,35% dan 6,51%. Bukan berarti tidak penting, hanya saja dengan jumlah data 300 baris, sinyal dari fitur-fitur ini belum cukup kuat untuk mendominasi keputusan model. Dengan lebih banyak data dari event-event berikutnya, kontribusi fitur-fitur ini kemungkinan akan meningkat dan memberikan gambaran prediksi yang lebih kaya dan lebih akurat.

---

*Dokumen ini dibuat berdasarkan kode program dan data aktual dari repositori skripsi. Seluruh angka yang tercantum berasal dari file metrics_summary.json dan split_info.json yang dihasilkan oleh proses pelatihan model yang sesungguhnya.*
