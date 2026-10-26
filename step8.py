import faiss
import pickle
import time
import csv
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai

UKURAN_CHUNK = 512
K = 3
NAMA_MODEL_LLM = "gemini-3.5-flash-lite"
JEDA_DETIK = 10

NAMA_FILE_HASIL = f"hasil_jawaban_{NAMA_MODEL_LLM}_{UKURAN_CHUNK}token.csv"
KOLOM = ["no", "pertanyaan", "jawaban_acuan", "jawaban_sistem", "sumber_terambil",
         "waktu_respons", "token_prompt", "tipe_pertanyaan", "model_llm", "ukuran_chunk"]

print("Memuat model embedding...")
model = SentenceTransformer("intfloat/multilingual-e5-large")
client = genai.Client()

index = faiss.read_index(f"index_{UKURAN_CHUNK}token.faiss")
with open(f"chunks_{UKURAN_CHUNK}token.pkl", "rb") as f:
    potongan = pickle.load(f)


def tanya_rag(pertanyaan):
    mulai = time.time()

    query = "query: " + pertanyaan
    angka_query = model.encode([query], normalize_embeddings=True)
    skor, indeks = index.search(np.array(angka_query, dtype=np.float32), k=K)

    konteks = ""
    sumber = []
    for idx in indeks[0]:
        chunk = potongan[idx]
        sumber.append(f"{chunk['dokumen']} hal {chunk['halaman']}")
        konteks += f"[{chunk['dokumen']}, Halaman {chunk['halaman']}]\n{chunk['teks']}\n\n"

    prompt = f"""Kamu adalah asisten yang membantu caregiver merawat penderita demensia.
Jawab pertanyaan HANYA berdasarkan konteks dokumen di bawah ini.
Jika jawabannya tidak ada di dalam konteks, katakan dengan jujur bahwa informasinya tidak tersedia di dokumen ini.

Konteks dokumen:
{konteks}

Pertanyaan: {pertanyaan}

Jawaban:"""

    respons = client.models.generate_content(model=NAMA_MODEL_LLM, contents=prompt)
    teks_jawaban = respons.text
    if not teks_jawaban:
        raise ValueError("Jawaban kosong dari model")
    durasi = time.time() - mulai

    return {
        "jawaban": teks_jawaban,
        "waktu_respons": round(durasi, 2),
        "token_prompt": respons.usage_metadata.prompt_token_count,
        "sumber_terambil": "; ".join(sumber),
    }


with open("pertanyaan_uji_final.csv", encoding="utf-8") as f:
    daftar_pertanyaan = list(csv.DictReader(f))

hasil_tersimpan = {}
if os.path.exists(NAMA_FILE_HASIL):
    with open(NAMA_FILE_HASIL, encoding="utf-8") as f:
        for baris in csv.DictReader(f):
            hasil_tersimpan[baris["no"]] = baris
print(f"Sudah ada {len(hasil_tersimpan)} jawaban tersimpan, sisanya akan dikerjakan.")


def simpan_semua():
    with open(NAMA_FILE_HASIL, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KOLOM)
        writer.writeheader()
        for no in sorted(hasil_tersimpan, key=int):
            writer.writerow(hasil_tersimpan[no])


berhenti_dini = False
for i, item in enumerate(daftar_pertanyaan, start=1):
    if item["no"] in hasil_tersimpan:
        continue

    pertanyaan = item["pertanyaan"]
    print(f"[{i}/{len(daftar_pertanyaan)}] {pertanyaan[:60]}...")

    hasil = None
    for percobaan in range(4):
        try:
            hasil = tanya_rag(pertanyaan)
            break
        except Exception as e:
            print("  Error:", str(e)[:100], "| coba lagi dalam 30 detik")
            time.sleep(30)

    if hasil is None:
        print("Pertanyaan ini gagal 4 kali. Kemungkinan jatah habis. Berhenti dulu, hasil sementara aman.")
        berhenti_dini = True
        break

    hasil_tersimpan[item["no"]] = {
        "no": item["no"],
        "pertanyaan": pertanyaan,
        "jawaban_acuan": item["jawaban_acuan"],
        "jawaban_sistem": hasil["jawaban"],
        "sumber_terambil": hasil["sumber_terambil"],
        "waktu_respons": hasil["waktu_respons"],
        "token_prompt": hasil["token_prompt"],
        "tipe_pertanyaan": item["tipe_pertanyaan"],
        "model_llm": NAMA_MODEL_LLM,
        "ukuran_chunk": UKURAN_CHUNK,
    }
    simpan_semua()
    time.sleep(JEDA_DETIK)

if berhenti_dini:
    print(f"Berhenti di tengah jalan. Tersimpan {len(hasil_tersimpan)} dari {len(daftar_pertanyaan)} jawaban. Jalankan ulang nanti untuk melanjutkan.")
else:
    print(f"Selesai, {len(hasil_tersimpan)} jawaban tersimpan di {NAMA_FILE_HASIL}")