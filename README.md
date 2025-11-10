# 🗑️ TrashClassifier — Image Classification untuk Jenis Sampah Menggunakan ML.NET

TrashClassifier adalah aplikasi machine learning berbasis **ML.NET** yang digunakan untuk **mendeteksi jenis sampah dari gambar**.  
Model dilatih menggunakan arsitektur **ResNet50** dan dapat mengklasifikasikan sampah ke dalam beberapa kategori, seperti:

- **Organik**
- **Anorganik**
- **B3 (Bahan Berbahaya & Beracun)**
- **Kertas**

---

## 📂 Struktur Dataset

Dataset yang digunakan diurutkan berdasarkan folder kelas:

dataset/
├── organik/
│ ├── img1.jpg
│ ├── img2.jpg
├── anorganik/
│ ├── img1.jpg
├── b3/
│ ├── img1.jpg
├── kertas/
│ ├── img1.jpg

yaml
Salin kode

> Pastikan nama folder = nama label kelas.

---

## 🏗️ Teknologi yang Digunakan

| Komponen | Teknologi |
|--------|-----------|
| Bahasa pemrograman | C# (.NET 8) |
| Framework ML | Microsoft ML.NET |
| Model pretrained | ResNet50 (Transfer Learning) |
| Library TensorFlow | SciSharp TensorFlow Redist |

---

## 🚀 Cara Menjalankan Proyek

### 1️⃣ Clone repository

git clone https://github.com/AyyMhmd/TrashClassifier.git
cd TrashClassifier
2️⃣ Pastikan .NET SDK 8 sudah terinstall
Cek versi:


dotnet --version
3️⃣ Restore & build

dotnet restore
dotnet build
4️⃣ Jalankan program

dotnet run
Jika dataset sudah sesuai, program akan:

Membuat file CSV label dataset

Membagi data (80% train / 20% test)

Melatih model

Menyimpan model dalam .zip

Menampilkan contoh prediksi

🧠 Hasil Pelatihan (Contoh Output)
yaml
Salin kode
✅ Pelatihan selesai!

📈 METRIK EVALUASI MODEL:
Akurasi (Micro): 91,41%
Akurasi (Macro): 91,93%
Log Loss: 0,2643

💾 Model disimpan sebagai: trash_model.zip

🔍 Contoh Prediksi:
File: biological1.jpg
Prediksi: organik
Skor per kelas:
 - kertas: 96,75%
 - organik: 2,00%
 - anorganik: 1,00%
📦 Model Output
Model hasil pelatihan disimpan dalam file:

python
Salin kode
trash_model.zip
File ini dapat digunakan untuk:

Integrasi ke aplikasi desktop (WPF/WinForms)

Web API (ASP.NET Core)

Mobile (Xamarin / MAUI)

IoT / Edge AI

✨ Pengembangan Selanjutnya
 Buat aplikasi UI (WPF / MAUI) untuk input gambar

 Integrasi ke web API ASP.NET untuk pengujian via browser

 Tambah dataset untuk meningkatkan akurasi

 Deploy ke perangkat IoT untuk pemilahan sampah otomatis

👨‍💻 Author
Nama: Ayyub Muhammad
GitHub: https://github.com/AyyMhmd

📝 Lisensi
Proyek ini menggunakan lisensi MIT — bebas digunakan untuk kebutuhan pembelajaran maupun pengembangan sistem.
