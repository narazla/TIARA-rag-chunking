import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai

print("Memuat model embedding...")
model = SentenceTransformer("intfloat/multilingual-e5-large")

print("Membuka index yang sudah tersimpan...")
index = faiss.read_index("index_256token.faiss")

with open("chunks_256token.pkl", "rb") as f:
    potongan = pickle.load(f)

client = genai.Client()

pertanyaan = "apa saja gejala awal demensia"

pertanyaan_dengan_prefix = "query: " + pertanyaan
angka_pertanyaan = model.encode([pertanyaan_dengan_prefix], normalize_embeddings=True)
skor, indeks = index.search(np.array(angka_pertanyaan, dtype=np.float32), k=3)

konteks = ""
for i in range(3):
    nomor_chunk = indeks[0][i]
    chunk_terkait = potongan[nomor_chunk]
    konteks += f"[Halaman {chunk_terkait['halaman']}]\n{chunk_terkait['teks']}\n\n"

prompt = f"""Kamu adalah asisten yang membantu caregiver merawat penderita demensia.
Jawab pertanyaan HANYA berdasarkan konteks dokumen di bawah ini.
Jika jawabannya tidak ada di dalam konteks, katakan dengan jujur bahwa informasinya tidak tersedia di dokumen ini.

Konteks dokumen:
{konteks}

Pertanyaan: {pertanyaan}

Jawaban:"""

respons = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt
)

print("=" * 50)
print("JAWABAN:")
print("=" * 50)
print(respons.text)