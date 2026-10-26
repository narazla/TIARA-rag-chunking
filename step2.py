from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-large")

print("Model berhasil dimuat")
print("Panjang deretan angka per kalimat:", model.get_sentence_embedding_dimension())

contoh_kalimat = "query: apa saja gejala awal demensia"
angka_hasil = model.encode(contoh_kalimat)

print("Bentuk hasilnya:", angka_hasil.shape)
print("Sepuluh angka pertama:", angka_hasil[:10])

from sklearn.metrics.pairwise import cosine_similarity

kalimat_mirip = "passage: demensia ditandai dengan penurunan daya ingat"
kalimat_beda = "passage: mother is washing the dishes in the kitchen"

angka_query = model.encode(contoh_kalimat).reshape(1, -1)
angka_mirip = model.encode(kalimat_mirip).reshape(1, -1)
angka_beda = model.encode(kalimat_beda).reshape(1, -1)

skor_mirip = cosine_similarity(angka_query, angka_mirip)[0][0]
skor_beda = cosine_similarity(angka_query, angka_beda)[0][0]

print("Skor kemiripan dengan kalimat yang relevan:", skor_mirip)
print("Skor kemiripan dengan kalimat yang tidak relevan:", skor_beda)