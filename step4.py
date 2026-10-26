import faiss
import pickle
from sentence_transformers import SentenceTransformer

print("Memuat model...")
model = SentenceTransformer("intfloat/multilingual-e5-large")

print("Membuka index yang sudah tersimpan...")
index = faiss.read_index("index_256token.faiss")

with open("chunks_256token.pkl", "rb") as f:
    potongan = pickle.load(f)

print("Index berisi", index.ntotal, "chunk")
print("Jumlah data teks:", len(potongan))

import numpy as np

pertanyaan = "apa saja gejala awal demensia"
pertanyaan_dengan_prefix = "query: " + pertanyaan

angka_pertanyaan = model.encode([pertanyaan_dengan_prefix], normalize_embeddings=True)

skor, indeks = index.search(np.array(angka_pertanyaan, dtype=np.float32), k=3)

print(f"\nPertanyaan: '{pertanyaan}'\n")

for i in range(3):
    nomor_chunk = indeks[0][i]
    skor_kemiripan = skor[0][i]
    chunk_terkait = potongan[nomor_chunk]

    print(f"--- Hasil #{i+1} (skor: {skor_kemiripan:.4f}, halaman {chunk_terkait['halaman']}) ---")
    print(chunk_terkait["teks"][:300])
    print()