import csv

with open("pertanyaan_uji_final.csv", encoding="utf-8") as f:
    pembaca = csv.DictReader(f)
    kolom = pembaca.fieldnames
    baris = list(pembaca)

for b in baris:
    if b["no"] == "19":
        b["pertanyaan"] = "Bagaimana tingkat bukti diklasifikasikan dalam PNPK Demensia?"
        b["jawaban_acuan"] = "Tingkat bukti dalam PNPK Demensia diklasifikasikan menjadi Level 1 sampai Level 4, diadaptasi dari sistem yang digunakan SIGN dan guideline internasional lainnya. Semakin rendah angka level, semakin tinggi kualitas bukti."
        b["halaman_sumber"] = "10"
        b["tipe_pertanyaan"] = "Faktual"
        b["kategori"] = "Peringkat Bukti"

with open("pertanyaan_uji_final.csv", "w", newline="", encoding="utf-8") as f:
    penulis = csv.DictWriter(f, fieldnames=kolom)
    penulis.writeheader()
    penulis.writerows(baris)

nama_hasil = "hasil_jawaban_gemini-3.5-flash-lite_256token.csv"
with open(nama_hasil, encoding="utf-8") as f:
    pembaca = csv.DictReader(f)
    kolom_hasil = pembaca.fieldnames
    hasil = [b for b in pembaca if b["no"] != "19"]

with open(nama_hasil, "w", newline="", encoding="utf-8") as f:
    penulis = csv.DictWriter(f, fieldnames=kolom_hasil)
    penulis.writeheader()
    penulis.writerows(hasil)

print("Selesai. Jalankan python step8.py untuk membuat ulang jawaban nomor 19.")