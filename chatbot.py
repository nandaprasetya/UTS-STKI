from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import gradio as gr
import re
import os
from google import genai
from google.genai import errors

client = genai.Client(
    api_key="Pakai Key Sendiri Jangan Pakai Key Ku -Nanda"
)

# Inisialisasi
emb     = SentenceTransformer("intfloat/multilingual-e5-small")
koleksi = chromadb.PersistentClient(path="chroma_db/").get_collection("dokumen_kampus")

KURIKULUM = {
    "2022": "Buku Kurikulum PSTI FT Unud 2022.pdf",
    "2026": "Buku Kurikulum 2026.pdf",
}

ROMAN_KE_ANGKA = {
    "viii": "8", "vii": "7", "vi": "6", "v": "5",
    "iv":   "4", "iii": "3", "ii": "2", "i": "1",
}

KATA_BANDING = [
    "perbedaan", "perbandingan", "banding", "dibanding",
    "versus", " vs ", "bedanya", "berbeda", "dibandingkan",
    "bandingkan", "compare", "comparison",
]


# Ekstrak intent
def ekstrak_intent(pertanyaan: str) -> dict:
    p = pertanyaan.lower()
    tahun_disebut = [th for th in KURIKULUM if th in p]
    is_compare = len(tahun_disebut) >= 2 or any(k in p for k in KATA_BANDING)
    mode = "compare" if is_compare else "single"

    semester = None
    m = re.search(r"semester\s+(viii|vii|vi|iv|v|iii|ii|i)\b", p)
    if m:
        semester = ROMAN_KE_ANGKA[m.group(1)]
    else:
        m = re.search(r"(?:semester|smt)[^\d]*(\d)", p)
        if m:
            semester = m.group(1)

    return {
        "tahun_list": tahun_disebut,           # list semua tahun
        "tahun":      tahun_disebut[0] if tahun_disebut else None,  # tahun utama (single)
        "semester":   semester,
        "mode":       mode,
    }


# Bangun filter ChromaDB
def bangun_filter(intent: dict) -> dict | None:
    if intent["mode"] == "compare":
        tahun_valid = [th for th in intent["tahun_list"] if th in KURIKULUM]
        if len(tahun_valid) == 0:
            return None
        if len(tahun_valid) == 1:
            return None
        return {"$or": [{"sumber": {"$eq": KURIKULUM[th]}} for th in tahun_valid]}

    kondisi = []
    if intent["tahun"] and intent["tahun"] in KURIKULUM:
        kondisi.append({"sumber": {"$eq": KURIKULUM[intent["tahun"]]}})
    if intent["semester"]:
        kondisi.append({"semester": {"$eq": intent["semester"]}})

    if len(kondisi) == 0:
        return None
    if len(kondisi) == 1:
        return kondisi[0]
    return {"$and": kondisi}


# Post-filter
def post_filter(dok_list, meta_list, intent):
    if intent["mode"] == "compare" or not intent["semester"]:
        return dok_list, meta_list

    target = intent["semester"]
    dok_ok, meta_ok = [], []
    for doc, meta in zip(dok_list, meta_list):
        smt = meta.get("semester", "unknown")
        if smt == target or smt == "unknown":
            dok_ok.append(doc)
            meta_ok.append(meta)

    if not dok_ok:
        print("[DEBUG] post_filter: fallback ke semua")
        return dok_list, meta_list

    return dok_ok, meta_ok


# Bangun prompt
def bangun_prompt(pertanyaan, konteks, intent):
    info_smt = f"Semester {intent['semester']}" if intent["semester"] else "semua semester"

    if intent["mode"] == "compare":
        tahun_str = " dan ".join(intent["tahun_list"]) if intent["tahun_list"] \
                    else "semua kurikulum yang tersedia"
        return f"""Kamu adalah chatbot akademik Program Studi Teknologi Informasi, FT Universitas Udayana.

ATURAN WAJIB:
1. Jawab HANYA berdasarkan KONTEKS di bawah.
2. JANGAN gunakan pengetahuan di luar konteks.
3. Pengguna meminta PERBANDINGAN antara kurikulum: {tahun_str}.
   → Jelaskan perbedaan secara terstruktur (gunakan tabel atau poin per poin).
   → Sebutkan dari kurikulum mana setiap informasi berasal.
4. Jika salah satu kurikulum tidak ada di konteks, sebutkan terus terang.
5. JANGAN mengarang data.

KONTEKS (dari {tahun_str}):
{konteks}

PERTANYAAN: {pertanyaan}
JAWABAN:"""

    else:
        info_thn = f"Kurikulum {intent['tahun']}" if intent["tahun"] else "kurikulum yang tersedia"
        return f"""Kamu adalah chatbot akademik Program Studi Teknologi Informasi, FT Universitas Udayana.

ATURAN WAJIB:
1. Jawab HANYA berdasarkan KONTEKS di bawah.
2. JANGAN gunakan pengetahuan di luar konteks.
3. Pengguna bertanya tentang: {info_thn}, {info_smt}.
   → Tampilkan HANYA data untuk {info_smt}.
   → JANGAN mencampurkan data dari semester lain.
4. Jika data tidak ada di konteks, jawab:
   "Data {info_smt} tidak ditemukan dalam {info_thn}."
5. Untuk daftar mata kuliah: tampilkan Kode, Nama MK, dan SKS
   untuk SEMUA mata kuliah dalam konteks. Jangan ada yang terlewat.
6. JANGAN mengarang data.

KONTEKS ({info_thn} — {info_smt}):
{konteks}

PERTANYAAN: {pertanyaan}
JAWABAN:"""


# Fungsi chat utama
def chat(pertanyaan: str, riwayat: list) -> str:
    intent = ekstrak_intent(pertanyaan)
    print(f"\n[DEBUG] Pertanyaan : {pertanyaan}")
    print(f"[DEBUG] Intent     : {intent}")

    where = bangun_filter(intent)
    print(f"[DEBUG] Filter DB  : {where}")

    vec    = emb.encode([f"query: {pertanyaan}"]).tolist()
    params = dict(query_embeddings=vec, n_results=15)
    if where:
        params["where"] = where

    hasil    = koleksi.query(**params)
    dok_raw  = hasil["documents"][0]
    meta_raw = hasil["metadatas"][0]
    print(f"[DEBUG] Hasil DB   : {len(dok_raw)} chunk sebelum post-filter")

    dok_f, meta_f = post_filter(dok_raw, meta_raw, intent)
    print(f"[DEBUG] Setelah pf : {len(dok_f)} chunk")

    if intent["mode"] == "compare":
        pasangan = sorted(zip(dok_f, meta_f), key=lambda x: x[1].get("sumber", ""))
        dok_sorted  = [d for d, _ in pasangan]
        meta_sorted = [m for _, m in pasangan]
        konteks = "\n\n---\n\n".join(dok_sorted[:14])
    else:
        konteks = "\n\n---\n\n".join(dok_f[:10])

    sumber_set = list(set(m["sumber"] for m in meta_f))
    prompt     = bangun_prompt(pertanyaan, konteks, intent)

    response = client.models.generate_content(
        model="gemma-4-31b-it",
        contents=prompt,
        config={
            "temperature": 0.05,
        }
    )

    jawaban = response.text
    sumber_str = ", ".join(sumber_set) if sumber_set else "tidak diketahui"
    return f"{jawaban}\n\n📄 Sumber: {sumber_str}"


# Launch Gradio
gr.ChatInterface(
    fn          = chat,
    title       = "Chatbot Kampus — PSTI FT Unud",
    description = (
        "Contoh pertanyaan:\n"
        "• Apa saja mata kuliah Semester 1 Kurikulum 2022?\n"
        "• Apa perbedaan Kurikulum 2022 dan 2026?\n"
        "• Berapa SKS Basis Data di Kurikulum 2022?"
    ),
).launch()