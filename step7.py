import faiss
import pickle
import numpy as np
import csv
from sentence_transformers import SentenceTransformer

UKURAN_CHUNK = 512

print("Memuat model...")
model = SentenceTransformer("intfloat/multilingual-e5-large")

print(f"Membuka index {UKURAN_CHUNK} token...")
index = faiss.read_index(f"index_{UKURAN_CHUNK}token.faiss")
with open(f"chunks_{UKURAN_CHUNK}token.pkl", "rb") as f:
    potongan = pickle.load(f)

with open("pertanyaan_uji_final.csv", encoding="utf-8") as f:
    daftar_pertanyaan = list(csv.DictReader(f))

print("Jumlah pertanyaan uji:", len(daftar_pertanyaan))

jumlah_berhasil = 0
hasil_detail = []

for item in daftar_pertanyaan:
    pertanyaan = item["pertanyaan"]
    halaman_benar = int(item["halaman_sumber"])

    query = "query: " + pertanyaan
    angka_query = model.encode([query], normalize_embeddings=True)
    skor, indeks = index.search(np.array(angka_query, dtype=np.float32), k=3)

    dokumen_benar = item["dokumen_sumber"]
    halaman_ditemukan = [(potongan[idx]["dokumen"], potongan[idx]["halaman"]) for idx in indeks[0]]
    berhasil = any(d == dokumen_benar and abs(h - halaman_benar) <= 1 for d, h in halaman_ditemukan)

    if berhasil:
        jumlah_berhasil += 1

    hasil_detail.append({
        "pertanyaan": pertanyaan,
        "halaman_benar": halaman_benar,
        "halaman_ditemukan": halaman_ditemukan,
        "berhasil": berhasil
    })

hit_rate = jumlah_berhasil / len(daftar_pertanyaan) * 100
print(f"\nHit rate untuk {UKURAN_CHUNK} token: {hit_rate:.1f}% ({jumlah_berhasil}/{len(daftar_pertanyaan)})")

print("\nContoh yang GAGAL ditemukan:")
for h in hasil_detail:
    if not h["berhasil"]:
        print(f"  - '{h['pertanyaan'][:60]}...' (benar: hal {h['halaman_benar']}, ditemukan: {h['halaman_ditemukan']})")