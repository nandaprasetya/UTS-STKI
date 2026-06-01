"""
load_dokumen.py — Debug & verifikasi struktur PDF
===================================================
Jalankan ini SEBELUM indexing untuk memastikan:
1. Halaman mana yang berisi data semester
2. State machine semester bekerja dengan benar
3. Sample teks per semester sudah sesuai
"""

import fitz
import re
from collections import defaultdict

POLA_SEMESTER = re.compile(r"Semester\s+(\d|I{1,3}V?|V?I{0,3})\b", re.IGNORECASE)
ROMAN_KE_ANGKA = {
    "i": "1", "ii": "2", "iii": "3", "iv": "4",
    "v": "5", "vi": "6", "vii": "7", "viii": "8",
}

def normalisasi_semester(raw: str) -> str:
    r = raw.strip().lower()
    return ROMAN_KE_ANGKA.get(r, r)

def debug_pdf(path: str):
    doc = fitz.open(path)
    print(f"\n{'='*60}")
    print(f"File    : {path}")
    print(f"Halaman : {len(doc)}")
    print('='*60)

    semester_aktif  = "unknown"
    halaman_per_smt = defaultdict(list)

    for i, hal in enumerate(doc):
        teks = hal.get_text()
        semua = POLA_SEMESTER.findall(teks)
        if semua:
            semester_aktif = normalisasi_semester(semua[-1])

        halaman_per_smt[semester_aktif].append(i + 1)

        # Tampilkan halaman yang mengandung perubahan semester
        if semua:
            print(f"\n>>> Halaman {i+1} — ditemukan: {semua} → semester_aktif = {semester_aktif}")
            print(teks[:500])
            print("---")

    print("\n\n=== RINGKASAN: Distribusi halaman per semester ===")
    for smt in sorted(halaman_per_smt, key=lambda x: (x == "unknown", x)):
        pages = halaman_per_smt[smt]
        label = f"Semester {smt}" if smt != "unknown" else "unknown"
        print(f"  {label:12s}: halaman {pages[:5]}{'...' if len(pages) > 5 else ''} ({len(pages)} halaman)")

    print("\n=== SAMPLE TEKS per Semester ===")
    ditampilkan = set()
    semester_aktif = "unknown"
    for i, hal in enumerate(doc):
        teks = hal.get_text()
        semua = POLA_SEMESTER.findall(teks)
        if semua:
            semester_aktif = normalisasi_semester(semua[-1])
        if semester_aktif not in ditampilkan and semester_aktif != "unknown":
            ditampilkan.add(semester_aktif)
            print(f"\n--- Semester {semester_aktif} (halaman {i+1}) ---")
            print(teks[:600])

if __name__ == "__main__":
    debug_pdf("dokumen/Buku Kurikulum PSTI FT Unud 2022.pdf")
    # Uncomment untuk debug kurikulum 2026 juga:
    # debug_pdf("dokumen/Buku Kurikulum 2026.pdf")