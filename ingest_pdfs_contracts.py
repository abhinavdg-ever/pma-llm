import os
import re
import sys
import uuid
from pathlib import Path
from typing import Generator, List, Tuple

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
from pypdf import PdfReader

# ---------------------
# CONFIGURATION
# ---------------------

# Usage:
#   python ingest_pdfs.py contracts
#   python ingest_pdfs.py docs

collection = sys.argv[1] if len(sys.argv) > 1 else "contracts"

if collection == "docs":
    pdf_folder = Path("./pdfs")
elif collection == "contracts":
    pdf_folder = Path("./contracts")
else:
    print("❌ Unknown collection. Use 'docs' or 'contracts'")
    sys.exit(1)

if not pdf_folder.exists():
    print(f"❌ Folder {pdf_folder} does not exist")
    sys.exit(1)

# ---------------------
# CONSTANTS
# ---------------------

TARGET_CHARS = 900
SENTENCE_OVERLAP = 1
SECTION_PATTERN = re.compile(
    r"^(SECTION\s+[0-9A-Z\-\.]+.*|ARTICLE\s+[0-9A-Z]+.*|APPENDIX\s+[A-Z]+.*)$",
    re.IGNORECASE,
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# ---------------------
# HELPERS
# ---------------------


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_slug(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()


def infer_doc_type(filename: str) -> str:
    lower = filename.lower()
    if "clerk" in lower:
        return "clerks"
    if "walking" in lower or "foremen" in lower or "foreman" in lower:
        return "walking_bosses"
    return "longshore"


def iter_sections(reader: PdfReader) -> Generator[Tuple[str, int, int, str], None, None]:
    current_title = "Preface"
    buffer: List[str] = []
    section_start = 1
    last_page = 1

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        lines = [normalize_whitespace(line) for line in raw_text.splitlines()]

        for line in lines:
            if not line:
                continue
            if SECTION_PATTERN.match(line.upper()):
                if buffer:
                    yield current_title, section_start, last_page, " ".join(buffer)
                    buffer = []
                current_title = line
                section_start = page_num
            else:
                buffer.append(line)
        last_page = page_num

    if buffer:
        yield current_title, section_start, last_page, " ".join(buffer)


def chunk_section_text(text: str) -> List[str]:
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return []

    chunks: List[str] = []
    start = 0

    while start < len(sentences):
        end = start
        current_sentences: List[str] = []
        current_chars = 0

        while end < len(sentences) and (current_chars < TARGET_CHARS or not current_sentences):
            sentence = sentences[end]
            current_sentences.append(sentence)
            current_chars += len(sentence)
            end += 1

        chunk_text = " ".join(current_sentences)
        chunks.append(chunk_text)

        if end >= len(sentences):
            break

        start = max(end - SENTENCE_OVERLAP, start + 1)

    return chunks


# ---------------------
# INGESTION
# ---------------------

print(f"✅ Ingesting into collection: {collection}")
print(f"📁 Reading PDFs from: {pdf_folder.resolve()}")

files = [p for p in pdf_folder.iterdir() if p.suffix.lower() == ".pdf"]

if not files:
    print("⚠️  No PDF files found. Exiting.")
    sys.exit(0)

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Connect to Qdrant
client = QdrantClient("http://localhost:6333")

# Recreate collection with cosine distance
client.recreate_collection(
    collection_name=collection,
    vectors_config=VectorParams(size=384, distance="Cosine"),
)

total_chunks = 0

for pdf_path in files:
    print(f"\n📄 Processing {pdf_path.name}...")
    reader = PdfReader(str(pdf_path))
    doc_type = infer_doc_type(pdf_path.name)

    for section_title, start_page, end_page, section_text in iter_sections(reader):
        chunks = chunk_section_text(section_text)
        if not chunks:
            continue

        section_slug = normalize_slug(section_title)

        for chunk_index, chunk in enumerate(chunks, start=1):
            vector = model.encode(chunk).tolist()
            payload = {
                "text": chunk,
                "section_heading": section_title,
                "section_slug": section_slug,
                "doc_type": doc_type,
                "start_page": start_page,
                "end_page": end_page,
                "chunk_index": chunk_index,
                "source": pdf_path.name,
            }

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            )

            client.upsert(collection_name=collection, points=[point])
            total_chunks += 1

print(f"\n✅ Finished ingesting {len(files)} PDFs into '{collection}' collection ({total_chunks} chunks).")
