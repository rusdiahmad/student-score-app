🎓 Student Performance Predictor (Prediksi Performa Siswa)

Aplikasi berbasis Machine Learning ini bertujuan untuk memberikan analisis prediktif terhadap potensi nilai ujian akhir seorang siswa berdasarkan 19 faktor input yang meliputi aspek Akademik, Lingkungan, dan Personal.

Tujuan utama dari alat ini adalah memberikan peringatan dini (early warning system) kepada pihak sekolah, guru, atau orang tua, serta merekomendasikan intervensi spesifik yang dapat dilakukan untuk memaksimalkan potensi akademik siswa.

🚀 Tautan Aplikasi

Aplikasi ini telah di-deploy menggunakan Streamlit. Anda dapat mengakses dan mencobanya langsung melalui tautan berikut:

Akses Aplikasi Prediksi Skor Siswa

📊 Detail Model

Deskripsi

Nilai

Arsitektur Model

Light Gradient Boosting Machine (LightGBM)

Metrik Evaluasi Utama

Mean Absolute Error (MAE)

Akurasi Model (MAE)

0.76 (Rata-rata prediksi meleset 0.76 poin dari skor aktual)

Jumlah Fitur Input

19 Faktor

🛠️ Fitur Utama

Prediksi Skor Akurat: Memberikan perkiraan skor ujian akhir (0-100) berdasarkan profil siswa.

Analisis Faktor: Menggunakan 19 faktor input (seperti jam belajar, kehadiran, motivasi, kualitas guru, dll.) untuk mendapatkan gambaran holistik.

Rekomendasi Intervensi: Secara otomatis menghasilkan daftar rekomendasi spesifik (misalnya: "Tingkatkan Jam Belajar menjadi 25 jam/minggu" atau "Pertimbangkan sesi bimbingan tambahan") berdasarkan kelemahan yang terdeteksi pada profil siswa.

💻 Cara Menggunakan Aplikasi

Akses aplikasi melalui tautan di atas.

Isi ke-19 kolom input (terbagi menjadi Akademik, Lingkungan, dan Personal) sesuai dengan data siswa yang ingin diprediksi.

Klik tombol "🔍 Analisis & Prediksi".

Aplikasi akan menampilkan hasil prediksi skor dan rekomendasi tindakan yang harus diprioritaskan.

Proyek ini dikembangkan menggunakan Python, Streamlit, dan LightGBM sebagai bagian dari inisiatif pengembangan sistem pendidikan berbasis data.
