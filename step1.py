from pypdf import PdfReader

def extract_text_with_pages(pdf_path):
    reader = PdfReader(pdf_path)
    semua_teks = ""
    batas_halaman = []
    for nomor_halaman, halaman in enumerate(reader.pages, start=1):
        teks_halaman = halaman.extract_text()
        posisi_mulai = len(semua_teks)
        semua_teks += teks_halaman
        posisi_selesai = len(semua_teks)
        batas_halaman.append((posisi_mulai, posisi_selesai, nomor_halaman))
    return semua_teks, batas_halaman


def extract_text_from_ocr(txt_path):
    with open(txt_path, encoding="utf-8") as f:
        isi = f.read()

    bagian = isi.split("=====HALAMAN ")[1:]
    semua_teks = ""
    batas_halaman = []
    for b in bagian:
        nomor_str, sisa = b.split("=====\n", 1)
        nomor_halaman = int(nomor_str.strip())
        posisi_mulai = len(semua_teks)
        semua_teks += sisa
        posisi_selesai = len(semua_teks)
        batas_halaman.append((posisi_mulai, posisi_selesai, nomor_halaman))
    return semua_teks, batas_halaman


def cari_nomor_halaman(posisi, batas_halaman):
    for posisi_mulai, posisi_selesai, nomor_halaman in batas_halaman:
        if posisi_mulai <= posisi < posisi_selesai:
            return nomor_halaman
    return batas_halaman[-1][2]


def potong_satu_dokumen(semua_teks, batas_halaman, nama_dokumen, ukuran_token, overlap_ratio=0.2):
    ukuran_karakter = ukuran_token * 4
    overlap_karakter = int(ukuran_karakter * overlap_ratio)
    langkah = ukuran_karakter - overlap_karakter

    hasil_potongan = []
    posisi = 0
    while posisi < len(semua_teks):
        teks_potongan = semua_teks[posisi:posisi + ukuran_karakter]
        halaman_potongan = cari_nomor_halaman(posisi, batas_halaman)
        hasil_potongan.append({
            "teks": teks_potongan,
            "halaman": halaman_potongan,
            "dokumen": nama_dokumen
        })
        posisi += langkah
    return hasil_potongan


def potong_teks(ukuran_token, overlap_ratio=0.2):
    teks_pnpk, batas_pnpk = extract_text_with_pages("pedoman nasional pelayanan klinis dementia.pdf")
    teks_caregiver, batas_caregiver = extract_text_from_ocr("panduan_caregiver_ocr.txt")

    potongan_pnpk = potong_satu_dokumen(teks_pnpk, batas_pnpk, "PNPK Demensia", ukuran_token, overlap_ratio)
    potongan_caregiver = potong_satu_dokumen(teks_caregiver, batas_caregiver, "Panduan Caregiver Kemenkes", ukuran_token, overlap_ratio)

    return potongan_pnpk + potongan_caregiver