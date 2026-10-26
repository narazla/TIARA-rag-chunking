import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from step1 import potong_teks

UKURAN_CHUNK = 512
print("Memuat model...")
model = SentenceTransformer("intfloat/multilingual-e5-large")

print(f"Memotong dokumen jadi {UKURAN_CHUNK} token...")
potongan = potong_teks(UKURAN_CHUNK)
print("Jumlah potongan:", len(potongan))

teks_dengan_prefix = ["passage: " + p["teks"] for p in potongan]

print("Mengubah semua potongan jadi angka, ini yang paling lama...")
semua_angka = model.encode(teks_dengan_prefix, show_progress_bar=True, normalize_embeddings=True)

print("Bentuk hasil akhirnya:", semua_angka.shape)

dimensi = semua_angka.shape[1]
index = faiss.IndexFlatIP(dimensi)
index.add(np.array(semua_angka, dtype=np.float32))

print("Index FAISS berhasil dibuat, isinya:", index.ntotal, "chunk")

faiss.write_index(index, f"index_{UKURAN_CHUNK}token.faiss")
print("Index berhasil disimpan ke file")

import pickle

with open(f"chunks_{UKURAN_CHUNK}token.pkl", "wb") as f:
    pickle.dump(potongan, f)

print("Data chunk berhasil disimpan")