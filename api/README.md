# MotoSense API

API ini digunakan untuk klasifikasi kerusakan mesin motor dari rekaman audio menggunakan model **YAMNet + Sequential TFLite**. Dokumentasi ini dibuat untuk memudahkan tim Frontend dalam melakukan integrasi.

## Base URL
Saat dijalankan di lokal, secara default API akan berada di:
`http://localhost:8000`

*CORS sudah diaktifkan (`allow_origins=["*"]`), sehingga Anda tidak akan mengalami kendala CORS saat melakukan request dari web frontend.*

---

## Endpoints

### 1. Cek Status (Health Check)
Mengecek apakah server berjalan dan memastikan model machine learning telah berhasil dimuat ke dalam memori.

- **URL:** `/`
- **Method:** `GET`
- **Response Sukses (200 OK):**
```json
{
  "status": "ready",
  "classes": [
    "Clutch-Shoe",
    "Conecting-Rod",
    "Drive-Belt",
    "Piston",
    "Tensioner",
    "Slider",
    "Roller",
    "Face-Drive"
  ],
  "model": "YAMNet + Sequential TFLite"
}
```
*(Catatan: Jika server baru saja dinyalakan dan model sedang dimuat, nilai `status` bisa berupa `"loading"`. Harap tunggu hingga `"ready"` sebelum melakukan prediksi).*

### 2. Daftar Kelas
Mengambil daftar komponen yang bisa diprediksi kerusakannya.

- **URL:** `/classes`
- **Method:** `GET`
- **Response Sukses (200 OK):**
```json
{
  "num_classes": 8,
  "classes": [
    "Clutch-Shoe",
    "Conecting-Rod",
    "Drive-Belt",
    "Piston",
    "Tensioner",
    "Slider",
    "Roller",
    "Face-Drive"
  ]
}
```

### 3. Prediksi Suara Mesin
Endpoint utama untuk mengirimkan rekaman audio dan mendapatkan prediksi komponen mesin mana yang rusak beserta nilai konfidensinya (confidence score).

- **URL:** `/predict`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Form Data (Body):**
  - Key: `file`
  - Value: File rekaman audio (format yang didukung: `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`)

**Contoh Response Sukses (200 OK):**
```json
{
  "filename": "rekaman_mesin_1.wav",
  "predicted_class": "Drive-Belt",
  "confidence": 0.9542,
  "all_scores": [
    {
      "label": "Clutch-Shoe",
      "probability": 0.0123
    },
    {
      "label": "Conecting-Rod",
      "probability": 0.0051
    },
    {
      "label": "Drive-Belt",
      "probability": 0.9542
    }
    // ... skor untuk kelas lainnya (total 8)
  ],
  "inference_ms": 145.2
}
```
**Penjelasan Field Response:**
- `predicted_class`: Komponen dengan probabilitas kerusakan tertinggi.
- `confidence`: Probabilitas dari class tertinggi (nilai 0.0 - 1.0).
- `all_scores`: Rincian skor probabilitas untuk setiap kelas (bisa digunakan untuk grafik di FE).
- `inference_ms`: Waktu pemrosesan model dalam milidetik.

**Response Gagal yang Mungkin Terjadi:**
- **400 Bad Request:** Format file tidak dikenali atau file kosong (ukurannya 0).
- **503 Service Unavailable:** Model belum selesai diinisialisasi saat start up. Harap tunggu sebentar lalu coba lagi.
- **500 Internal Server Error:** Gagal memproses file audio (misal file korup/rusak).

---

## Test Interaktif (Swagger)
FastAPI juga secara otomatis membuatkan halaman dokumentasi interaktif. Anda bisa langsung mencoba upload file tanpa perlu menulis kode melalui Swagger UI:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**
