import sys
import uuid
from pathlib import Path
from typing import Generator

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct

from parse_contract import (
    extract_raw_text,
    clean_text,
    parse_contract,
    combine_clauses_by_main,
)

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

CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 100  # characters

# ---------------------
# HELPERS
# ---------------------


def infer_doc_type(filename: str) -> str:
    lower = filename.lower()
    if "clerk" in lower:
        return "clerks"
    if "walking" in lower or "foremen" in lower or "foreman" in lower:
        return "walking_bosses"
    return "longshore"


def iter_char_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Generator[str, None, None]:
    """Split text into character-based chunks with overlap."""
    if not text:
        return
    
    text_len = len(text)
    start = 0
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # Only yield non-empty chunks
            yield chunk
        start = end - overlap  # Move forward by chunk_size - overlap


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
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Connect to Qdrant
client = QdrantClient("http://localhost:6333")

# Recreate collection with cosine distance
# Note: all-mpnet-base-v2 produces 768-dimensional vectors
client.recreate_collection(
    collection_name=collection,
    vectors_config=VectorParams(size=768, distance="Cosine"),
)

total_chunks = 0

for pdf_path in files:
    print(f"\n📄 Processing {pdf_path.name}...")
    doc_type = infer_doc_type(pdf_path.name)
    
    # Extract and parse contract using parse_contract.py
    try:
        raw_text = extract_raw_text(pdf_path)
        cleaned_text = clean_text(raw_text)
        sections, clauses = parse_contract(cleaned_text, raw_text=raw_text)
        
        if not sections or not clauses:
            print(f"⚠️  No sections or clauses found in {pdf_path.name}. Skipping.")
            continue
        
        # Get combined clauses (section_clauses_combined format)
        combined_rows = combine_clauses_by_main(sections, clauses)
        
        print(f"   Found {len(sections)} sections, {len(clauses)} clauses, {len(combined_rows)} combined groups")
        
        # Create section map for efficient lookup
        section_map = {s["section_id"]: s for s in sections}
        
        # Process each combined row
        for row in combined_rows:
            base_text = row.get("text", "").strip()
            if not base_text:
                continue
            
            # Chunk the text with character-based chunking and overlap
            char_chunks = list(iter_char_chunks(base_text, CHUNK_SIZE, CHUNK_OVERLAP))
            if not char_chunks:
                continue
            
            # Extract metadata from combined row
            section_id = row.get("section_id", "")
            section_title = row.get("section_title", "")
            main_clause_id = row.get("main_clause_id", "")
            sub_clause_ids = row.get("sub_clause_ids", "")
            
            # Get page numbers from sections
            section_info = section_map.get(section_id, {})
            start_page = section_info.get("start_page", "")
            end_page = section_info.get("end_page", "")
            
            # Create embeddings for each chunk
            for chunk_index, chunk in enumerate(char_chunks, start=1):
                vector = model.encode(chunk).tolist()
                payload = {
                    "text": chunk,
                    "section_id": section_id,
                    "section_title": section_title,
                    "main_clause_id": main_clause_id,
                    "sub_clause_ids": sub_clause_ids,
                    "start_page": start_page,
                    "end_page": end_page,
                    "doc_type": doc_type,
                    "source": pdf_path.name,
                    "chunk_part": chunk_index,
                }
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload,
                )
                
                client.upsert(collection_name=collection, points=[point])
                total_chunks += 1
                
    except Exception as e:
        print(f"❌ Error processing {pdf_path.name}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n✅ Finished ingesting {len(files)} PDFs into '{collection}' collection ({total_chunks} chunks).")
