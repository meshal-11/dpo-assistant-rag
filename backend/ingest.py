import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import os
import re
import json
import hashlib
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pymupdf

from backend.config import (
    PDF_PATH,
    CACHE_PATH,
    OPENAI_EMBEDDING_MODEL,
    CHUNKING_VERSION,
    get_openai_client,
    is_openai_configured,
)

EXCLUDED_PAGES = {1, 2, 3, 11}  # Page 1: Cover, Page 2: Basmala, Page 3: TOC, Page 11: Back cover


def compute_config_hash(pdf_path: Path, chunk_version: str, embedding_model: str) -> str:
    """Compute a combined SHA256 hash of the PDF binary, chunking config, and embedding model."""
    hasher = hashlib.sha256()
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    else:
        hasher.update(b"NO_PDF_FILE")
    hasher.update(chunk_version.encode("utf-8"))
    hasher.update(embedding_model.encode("utf-8"))
    return hasher.hexdigest()


def remove_tatweel(text: str) -> str:
    """Remove Arabic kashida/tatweel character."""
    return re.sub(r'[\u0640]', '', text)


def fix_reversed_arabic_segments(text: str) -> str:
    """
    Fix visually reversed Arabic lines identified in the PDF extraction on page 4.
    PyMuPDF extracted standard forward text on other pages, but on page 4, certain
    decree citations were encoded in reverse visual character order in the original PDF.
    """
    replacements = [
        ("ةداملا نوثلاثلا نم ماظن ةيامح تانايبلا ةيصخشلا رداصلا موسرملاب يكلملا مقر )٩١ /م( خيراتو",
         "المادة الثلاثون من نظام حماية البيانات الشخصية الصادر بالمرسوم الملكي رقم (م/١٩) وتاريخ"),
        ("موسرملاب يكلملا مقر )٩١ /م( خيراتو ٩/٢/٣٤٤١",
         "الصادر بالمرسوم الملكي رقم (م/١٩) وتاريخ ١٤٤٣/٢/٩"),
        ("ةداملا نوثلاثلا نم ماظن ةيامح تانايبلا ةيصخشلا",
         "المادة الثلاثون من نظام حماية البيانات الشخصية"),
        ("موسرملاب يكلملا مقر", "بالمرسوم الملكي رقم"),
        ("٩/٢/٣٤٤١", "١٤٤٣/٢/٩"),
        ("٨٤١( خيراتو ٥/٩/٤٤٤١", "(١٤٨) وتاريخ ١٤٤٤/٩/٥"),
        ("٨٤١(", "(١٤٨)"),
        ("٥/٩/٤٤٤١", "١٤٤٤/٩/٥"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def clean_page_text(raw_text: str, page_number: int) -> str:
    """Clean and normalize extracted page text."""
    # 1. Unicode NFKC normalization
    text = unicodedata.normalize('NFKC', raw_text)

    # 2. Remove repeated header / footer lines
    text = re.sub(r'^\s*\d+\s+تصنيف الوثيقة:\s*عام\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*تصنيف الوثيقة:\s*عام\s*\d*\s*', '', text, flags=re.MULTILINE)

    # 3. Remove tatweel
    text = remove_tatweel(text)

    # 4. Fix reversed segments on Page 4 (or wherever detected)
    if page_number == 4 or "ةداملا نوثلاثلا" in text:
        text = fix_reversed_arabic_segments(text)

    # 5. Remove layout artifact exclamation marks
    text = re.sub(r'!+', '', text)

    # 6. Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()


def extract_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract and clean pages from the PDF."""
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"ملف الوثيقة غير موجود في المسار: {pdf_path}\n"
            f"PDF file not found at: {pdf_path}"
        )

    doc = pymupdf.open(str(pdf_path))
    if len(doc) == 0:
        raise ValueError("ملف PDF فارغ أو تالف. / The PDF document is empty or corrupted.")

    pages = []
    for page_idx in range(len(doc)):
        page_number = page_idx + 1  # 1-based page number
        if page_number in EXCLUDED_PAGES:
            continue

        raw_text = doc[page_idx].get_text()
        cleaned = clean_page_text(raw_text, page_number)

        # Skip effectively empty pages
        if len(cleaned) < 25:
            continue

        pages.append({
            "page_number": page_number,
            "text": cleaned
        })

    if not pages:
        raise ValueError("فشل استخراج أي نصوص صالحة من الوثيقة. / Failed to extract any valid text from PDF.")

    return pages


def chunk_extracted_pages(pages: List[Dict[str, Any]], max_chars: int = 480, overlap_chars: int = 80) -> List[Dict[str, Any]]:
    """
    Split pages into structured, coherent chunks while preserving exact 1-based page numbers.
    Articles and bullet sections are maintained together when possible.
    """
    chunks = []
    chunk_counter = 1

    for p in pages:
        page_number = p["page_number"]
        page_text = p["text"]

        raw_lines = [l.strip() for l in page_text.split('\n') if l.strip()]

        # Group lines into semantic blocks (Articles, Section headings, numbered paragraphs)
        blocks = []
        current_block = ""
        current_heading = f"صفحة {page_number}"

        for line in raw_lines:
            is_new_section = any(line.startswith(prefix) for prefix in [
                "المادة", "المقدمة", "أولاً:", "ثانياً:", "ثالثاً:", "رابعاً:", "خامساً:"
            ])

            if is_new_section and current_block:
                blocks.append(current_block)
                current_block = line
            else:
                if current_block:
                    current_block += " " + line
                else:
                    current_block = line

        if current_block:
            blocks.append(current_block)

        # Sub-split blocks if they exceed max_chars
        for block in blocks:
            if len(block) <= max_chars:
                chunks.append({
                    "chunk_id": f"chunk_{chunk_counter:02d}",
                    "page_number": page_number,
                    "text": block.strip()
                })
                chunk_counter += 1
            else:
                sentences = re.split(r'(?<=[.؛؟\n])\s+', block)
                sub_chunk = ""
                for s in sentences:
                    if not sub_chunk:
                        sub_chunk = s
                    elif len(sub_chunk) + len(s) + 1 <= max_chars:
                        sub_chunk += " " + s
                    else:
                        chunks.append({
                            "chunk_id": f"chunk_{chunk_counter:02d}",
                            "page_number": page_number,
                            "text": sub_chunk.strip()
                        })
                        chunk_counter += 1

                        # Carry forward overlap
                        words = sub_chunk.split()
                        overlap = " ".join(words[-12:]) if len(words) > 12 else sub_chunk
                        sub_chunk = overlap + " " + s

                if sub_chunk and sub_chunk.strip():
                    chunks.append({
                        "chunk_id": f"chunk_{chunk_counter:02d}",
                        "page_number": page_number,
                        "text": sub_chunk.strip()
                    })
                    chunk_counter += 1

    return chunks


def build_or_load_embeddings_cache(
    pdf_path: Path = PDF_PATH,
    cache_path: Path = CACHE_PATH,
    embedding_model: str = OPENAI_EMBEDDING_MODEL,
    force_rebuild: bool = False
) -> Tuple[List[Dict[str, Any]], List[List[float]], str]:
    """
    Build or load document embeddings from local cache.
    Rebuilds the cache when the document binary, embedding model, or chunking configuration changes.
    """
    current_hash = compute_config_hash(pdf_path, CHUNKING_VERSION, embedding_model)

    # 1. Try loading from cache
    if not force_rebuild and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)

            if cached_data.get("config_hash") == current_hash and "chunks" in cached_data and "embeddings" in cached_data:
                print(f"[Cache] Successfully loaded {len(cached_data['chunks'])} chunks from cache: {cache_path}")
                return cached_data["chunks"], cached_data["embeddings"], current_hash
            else:
                print("[Cache] Hash mismatch or cache schema outdated. Rebuilding cache...")
        except Exception as e:
            print(f"[Cache] Error reading cache ({e}). Rebuilding...")

    # 2. Extract pages and create chunks
    print(f"[Ingest] Extracting text from PDF: {pdf_path}")
    pages = extract_pages(pdf_path)
    chunks = chunk_extracted_pages(pages)
    print(f"[Ingest] Generated {len(chunks)} chunks across {len(pages)} pages.")

    # 3. Generate embeddings using OpenAI API
    if not is_openai_configured():
        print("[Ingest] OpenAI API key is not configured. Chunks extracted, but embeddings pending API key.")
        return chunks, [], current_hash

    client = get_openai_client()
    texts = [c["text"] for c in chunks]
    print(f"[Ingest] Requesting embeddings from OpenAI using model '{embedding_model}' for {len(texts)} chunks...")

    response = client.embeddings.create(
        model=embedding_model,
        input=texts
    )
    embeddings = [item.embedding for item in response.data]

    # 4. Save to local cache
    cache_data = {
        "config_hash": current_hash,
        "chunking_version": CHUNKING_VERSION,
        "embedding_model": embedding_model,
        "chunks_count": len(chunks),
        "chunks": chunks,
        "embeddings": embeddings
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)

    print(f"[Cache] Saved embeddings cache to: {cache_path}")
    return chunks, embeddings, current_hash
