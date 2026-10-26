# Analisis Pengaruh Ukuran Chunk terhadap Kualitas Jawaban Sistem RAG pada Chatbot Pendamping Caregiver Demensia

Repositori ini berisi kode, data, dan hasil eksperimen untuk penelitian skripsi yang membandingkan tiga ukuran chunk (128, 256, dan 512 token) pada sistem Retrieval-Augmented Generation (RAG) untuk chatbot pendamping caregiver demensia berbahasa Indonesia.

## Ringkasan Penelitian

Chatbot pendamping caregiver demensia membutuhkan jawaban yang berpijak pada dokumen panduan resmi, bukan pengetahuan umum model bahasa. Salah satu komponen penting dalam sistem RAG adalah strategi chunking, cara dokumen dipotong menjadi bagian-bagian kecil sebelum diproses. Ukuran potongan ini berpengaruh langsung terhadap kualitas retrieval dan jawaban akhir.

Penelitian ini mengevaluasi tiga konfigurasi ukuran chunk pada dua lapis evaluasi: kualitas retrieval dengan metrik hit rate, dan kualitas jawaban akhir yang akan dievaluasi menggunakan ROUGE dan BERTScore.

## Dokumen Sumber

Penelitian menggunakan dua dokumen resmi Kementerian Kesehatan Republik Indonesia.

- **Pedoman Nasional Pelayanan Klinis Tata Laksana Demensia** (PNPK Demensia), 204 halaman
- **Panduan Praktis Caregiver dalam Perawatan Jangka Panjang bagi Lanjut Usia** (Panduan Caregiver Kemenkes), hasil OCR

## Konfigurasi Eksperimen

| Variabel | Nilai |
|---|---|
| Model embedding | multilingual-e5-large (dimensi 1024) |
| Ukuran chunk | 128, 256, 512 token |
| Overlap | 20% |
| Vector index | FAISS IndexFlatIP (inner product) |
| Jumlah chunk teratas (K) | 3 |
| LLM | Gemini 3.5 Flash Lite |
| Jumlah pertanyaan uji | 34 |

## Hasil Evaluasi Retrieval (Hit Rate)

Evaluasi dilakukan pada 34 pertanyaan uji dengan toleransi selisih satu halaman.

| Ukuran Chunk | Jumlah Chunk | Hit Rate |
|---|---|---|
| 128 token | 1114 | 91,2% |
| **256 token** | **558** | **97,1%** |
| 512 token | 279 | 91,2% |

Hasil tidak linear. Ukuran chunk di tengah mengungguli ukuran terkecil maupun terbesar.

## Struktur Repositori

- `step1.py` sampai `step8.py` — pipeline eksperimen tahap 1 sampai 8
- `perbaiki_19.py` — skrip perbaikan pertanyaan uji nomor 19
- `pertanyaan_uji_final.csv` — set uji 34 pertanyaan dengan anotasi ground truth
- `chunks_128token.pkl` — data chunk ukuran 128 token (1114 chunk)
- `chunks_256token.pkl` — data chunk ukuran 256 token (558 chunk)
- `chunks_512token.pkl` — data chunk ukuran 512 token (279 chunk)
- `index_128token.faiss` — FAISS index embedding 128 token
- `index_256token.faiss` — FAISS index embedding 256 token
- `index_512token.faiss` — FAISS index embedding 512 token
- `hasil_jawaban_gemini-3.5-flash-lite_128token.csv` — hasil jawaban sistem konfigurasi 128 token
- `hasil_jawaban_gemini-3.5-flash-lite_256token.csv` — hasil jawaban sistem konfigurasi 256 token
- `hasil_jawaban_gemini-3.5-flash-lite_512token.csv` — hasil jawaban sistem konfigurasi 512 token
- `dementia_guidelines.txt` — teks hasil ekstraksi PDF PNPK Demensia
- `panduan_caregiver_ocr.txt` — teks hasil OCR Panduan Caregiver Kemenkes
- `pedoman nasional pelayanan klinis dementia.pdf` — PDF asli PNPK Kemenkes

## Alur Pipeline

1. `step1.py` — ekstraksi teks dari PDF PNPK Demensia menggunakan pypdf
2. `step2.py` — chunking dengan metode fixed-size, overlap 20%, menyimpan metadata halaman sumber
3. `step3.py` — embedding dengan multilingual-e5-large
4. `step4.py` — penyimpanan embedding ke FAISS IndexFlatIP dan teks chunk ke pickle
5. `step5.py` — uji coba retrieval dan generation (RAG end-to-end)
6. `step6.py` — pembangunan index untuk tiga ukuran chunk
7. `step7.py` — evaluasi retrieval dengan metrik hit rate
8. `step8.py` — pembuatan jawaban sistem untuk evaluasi lapis dua

## Set Uji

File `pertanyaan_uji_final.csv` berisi 34 pertanyaan dengan kolom berikut.

- `no` — nomor pertanyaan
- `pertanyaan` — teks pertanyaan
- `jawaban_acuan` — jawaban acuan manual
- `halaman_sumber` — halaman sumber (ground truth)
- `dokumen_sumber` — nama dokumen sumber
- `tipe_pertanyaan` — Faktual, Deskriptif, atau Sintesis
- `kategori` — kategori topik (Diagnosis, Faktor Risiko, Terapi Farmakologi, Perawatan Harian, dan lain-lain)

## Cara Menjalankan Ulang

### Prasyarat

- Python 3.11
- API key Google Gemini (untuk step5 dan step8)

### Instalasi

```
python -m venv venv
venv\Scripts\Activate.ps1
pip install sentence-transformers faiss-cpu google-genai pypdf numpy pandas
```

### Menjalankan evaluasi retrieval

Buka `step7.py`, ganti variabel `UKURAN_CHUNK` ke 128, 256, atau 512, lalu jalankan.

```
python step7.py
```

### Menjalankan pembuatan jawaban sistem

Buka `step8.py`, ganti variabel `UKURAN_CHUNK` sesuai konfigurasi yang mau diuji, lalu jalankan.

```
python step8.py
```

Hasil disimpan ke `hasil_jawaban_gemini-3.5-flash-lite_{UKURAN_CHUNK}token.csv`.

## Lisensi dan Atribusi

Dokumen sumber adalah dokumen publik Kementerian Kesehatan Republik Indonesia dan digunakan untuk keperluan penelitian akademik. Hak cipta dokumen tetap pada Kemenkes.
