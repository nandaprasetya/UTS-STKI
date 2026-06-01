"""
indexing.py — Revisi dengan state-machine parser semester
=========================================================
Masalah sebelumnya: regex tidak menangkap semester karena
"Semester 1" hanya muncul sebagai header tabel, terpisah dari
baris-baris mata kuliah di halaman berikutnya.

Solusi: baca PDF halaman-per-halaman, lacak semester aktif
dengan state machine, lalu inject ke setiap chunk.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import fitz          # PyMuPDF
import docx
import os
import re
from collections import Counter


# ─── Baca file ────────────────────────────────────────────────────────────────

def baca_pdf_per_halaman(path: str) -> list[dict]:
    """
    Baca PDF halaman demi halaman.
    Return list of {"halaman": int, "teks": str}
    """
    doc = fitz.open(path)
    return [
        {"halaman": i + 1, "teks": hal.get_text()}
        for i, hal in enumerate(doc)
    ]

def baca_docx(path: str) -> str:
    return "\n".join([p.text for p in docx.Document(path).paragraphs])

def muat_semua(folder: str) -> list[dict]:
    hasil = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if file.endswith(".pdf"):
            halaman_list = baca_pdf_per_halaman(path)
            hasil.append({
                "nama": file,
                "tipe": "pdf",
                "halaman_list": halaman_list,
                "teks": "\n".join(h["teks"] for h in halaman_list),
            })
            total_kar = sum(len(h["teks"]) for h in halaman_list)
            print(f"Terbaca: {file} ({len(halaman_list)} halaman, {total_kar} karakter)")
        elif file.endswith(".docx"):
            teks = baca_docx(path)
            hasil.append({
                "nama": file,
                "tipe": "docx",
                "halaman_list": [{"halaman": 1, "teks": teks}],
                "teks": teks,
            })
            print(f"Terbaca: {file} ({len(teks)} karakter)")
    return hasil


# ─── Ekstrak tahun kurikulum dari nama file ────────────────────────────────────

def ekstrak_tahun_kurikulum(nama_file: str) -> str:
    m = re.search(r"(20\d{2})", nama_file)
    return m.group(1) if m else "unknown"


# ─── State machine: lacak semester aktif per halaman ──────────────────────────

POLA_SEMESTER = re.compile(
    r"Semester\s+(\d|I{1,3}V?|V?I{0,3})\b",
    re.IGNORECASE
)
ROMAN_KE_ANGKA = {
    "i": "1", "ii": "2", "iii": "3", "iv": "4",
    "v": "5", "vi": "6", "vii": "7", "viii": "8",
}

def normalisasi_semester(raw: str) -> str:
    """Ubah '1'/'I'/'II'/dll jadi string angka '1'..'8'."""
    r = raw.strip().lower()
    return ROMAN_KE_ANGKA.get(r, r)  # angka sudah jadi angka, romawi dikonversi

def buat_halaman_dengan_semester(halaman_list: list[dict]) -> list[dict]:
    """
    Tambahkan field 'semester_aktif' ke setiap halaman menggunakan state machine.
    Semester aktif mengikuti 'Semester N' terakhir yang terlihat.
    """
    semester_aktif = "unknown"
    hasil = []
    for hal in halaman_list:
        teks = hal["teks"]
        # Cari semua kemunculan "Semester N" di halaman ini (ambil yang terakhir jika >1)
        semua = POLA_SEMESTER.findall(teks)
        if semua:
            semester_aktif = normalisasi_semester(semua[-1])
        hasil.append({**hal, "semester_aktif": semester_aktif})
    return hasil


# ─── Buat chunks dengan metadata semester yang akurat ─────────────────────────

def buat_chunks(dokumen: list[dict]) -> list[dict]:
    """
    Chunking strategi ganda:
    1. Blok per-semester dibangun dari halaman, lalu di-split.
       Setiap chunk mewarisi semester_aktif dari halaman asalnya.
    2. chunk_size besar (1000) agar header tabel tidak terpisah dari isinya.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\nSemester ",   # pecah di header semester baru
            "\nNo\nKode",    # pecah di header tabel baru
            "\nBidang Minat",
            "\n\n",
            "\n",
            ".",
            " ",
        ],
    )

    chunks = []
    for dok in dokumen:
        tahun = ekstrak_tahun_kurikulum(dok["nama"])
        halaman_enriched = buat_halaman_dengan_semester(dok["halaman_list"])

        # Gabungkan teks per-semester (bukan per-halaman) agar context utuh
        # Key: semester_aktif → akumulasi teks
        blok_per_semester: dict[str, list[str]] = {}
        for hal in halaman_enriched:
            smt = hal["semester_aktif"]
            blok_per_semester.setdefault(smt, []).append(hal["teks"])

        chunk_id = 0
        for smt, teks_list in blok_per_semester.items():
            teks_gabung = "\n".join(teks_list)
            potongan_list = splitter.split_text(teks_gabung)
            for pot in potongan_list:
                # Verifikasi isi: pastikan chunk ini tidak lebih cocok dengan semester lain
                # (untuk chunk yang lintas-halaman sehingga mungkin mengandung 2 semester)
                semua_smt_dalam_chunk = POLA_SEMESTER.findall(pot)
                if semua_smt_dalam_chunk:
                    # Pakai semester terakhir yang disebutkan dalam chunk
                    smt_efektif = normalisasi_semester(semua_smt_dalam_chunk[-1])
                else:
                    smt_efektif = smt

                chunks.append({
                    "id":       f"{dok['nama']}_{chunk_id}",
                    "teks":     pot,
                    "sumber":   dok["nama"],
                    "chunk_id": chunk_id,
                    "semester": smt_efektif,
                    "tahun":    tahun,
                })
                chunk_id += 1

    return chunks


# ─── Simpan ke ChromaDB ───────────────────────────────────────────────────────

def simpan_ke_db(chunks: list[dict], koleksi) -> None:
    BATCH = 100  # hindari timeout pada koleksi besar
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        koleksi.add(
            ids        = [c["id"] for c in batch],
            embeddings = model.encode(
                [f"passage: {c['teks']}" for c in batch]
            ).tolist(),
            documents  = [c["teks"] for c in batch],
            metadatas  = [
                {
                    "sumber":   c["sumber"],
                    "chunk_id": c["chunk_id"],
                    "semester": c["semester"],
                    "tahun":    c["tahun"],
                }
                for c in batch
            ],
        )
        print(f"  Tersimpan batch {i//BATCH + 1}: chunk {i}–{i+len(batch)-1}")


# ─── Main ─────────────────────────────────────────────────────────────────────

model  = SentenceTransformer("intfloat/multilingual-e5-small")
client = chromadb.PersistentClient(path="chroma_db/")

try:
    client.delete_collection("dokumen_kampus")
    print("Collection lama dihapus")
except Exception:
    print("Collection belum ada, dibuat baru")

koleksi = client.create_collection("dokumen_kampus")

dokumen = muat_semua("dokumen/")
chunks  = buat_chunks(dokumen)
simpan_ke_db(chunks, koleksi)
print(f"\nIndexing selesai! Total: {len(chunks)} chunk")

# ─── Debug: distribusi chunk per (tahun, semester) ───────────────────────────
print("\nDistribusi chunk per (tahun, semester):")
dist = Counter((c["tahun"], c["semester"]) for c in chunks)
for (tahun, smt), jml in sorted(dist.items()):
    label = f"Semester {smt}" if smt != "unknown" else "unknown"
    print(f"  Tahun {tahun} | {label:12s} : {jml:3d} chunk")

# ─── Debug: cek 3 chunk pertama per semester untuk kurikulum 2022 ─────────────
print("\n\nSAMPLE CHUNK per Semester (Kurikulum 2022):")
print("=" * 60)
ditampilkan = set()
for c in chunks:
    key = (c["tahun"], c["semester"])
    if c["tahun"] == "2022" and key not in ditampilkan:
        ditampilkan.add(key)
        print(f"\n--- Tahun {c['tahun']} | Semester {c['semester']} ---")
        print(c["teks"][:400])
        print("...")