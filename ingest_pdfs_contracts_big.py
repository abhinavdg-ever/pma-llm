import os
import sys
import uuid
import time
from pathlib import Path
from typing import Generator, List, Optional

import requests
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
#   python ingest_pdfs_contracts_big.py contracts_big

collection = sys.argv[1] if len(sys.argv) > 1 else "contracts_big"

if collection == "contracts_big":
    pdf_folder = Path("./contracts")
else:
    print("❌ Unknown collection. Use 'contracts_big'")
    sys.exit(1)

if not pdf_folder.exists():
    print(f"❌ Folder {pdf_folder} does not exist")
    sys.exit(1)

# ---------------------
# CONSTANTS (HARDCODED FOR VM)
# ---------------------

# Embedding configuration - Set via environment variables
EMBEDDING_TYPE = os.getenv("EMBEDDING_TYPE", "openai")  # "sentence_transformers" or "openai"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set via environment variable
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dims
OPENAI_EMBEDDING_URL = "https://api.openai.com/v1/embeddings"

# Qdrant configuration - HARDCODED for VM (localhost since Qdrant is on same VM)
QDRANT_URL = "http://localhost:6333"  # Qdrant running on same VM

# Chunking configuration - DISABLED (vectorize at clause level)
CHUNK_SIZE = 0  # 0 = no chunking, each combined clause gets one vector
CHUNK_OVERLAP = 100  # Not used when chunking disabled

# Vector dimensions based on embedding type
VECTOR_DIMENSIONS = {
    "sentence_transformers": 768,  # all-mpnet-base-v2
    "openai": 1536,  # text-embedding-3-small (3072 for large)
}

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


def iter_char_chunks(text: str, chunk_size: Optional[int] = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Generator[str, None, None]:
    """Split text into character-based chunks with overlap. If chunk_size is 0 or None, returns text as-is."""
    if not text:
        return
    
    # If chunking is disabled, return entire text as single chunk
    if not chunk_size or chunk_size <= 0:
        yield text
        return
    
    text_len = len(text)
    start = 0
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # Only yield non-empty chunks
            yield chunk
        start = end - overlap  # Move forward by chunk_size - overlap


def embed_with_openai(text: str, max_retries: int = 5, initial_delay: float = 1.0) -> List[float]:
    """Generate embedding using OpenAI API with retry logic for rate limits."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in the script")
    
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OPENAI_EMBEDDING_URL,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_EMBEDDING_MODEL,
                    "input": text,
                },
                timeout=120,  # Increased timeout
            )
            
            # Check for rate limit or server errors
            if response.status_code == 429:
                # Rate limited - get retry-after header or use exponential backoff
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    delay = float(retry_after)
                else:
                    delay = min(delay * 2, 60)  # Cap at 60 seconds
                print(f"      Rate limited. Retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})", end="\r")
                time.sleep(delay)
                continue
            
            if response.status_code >= 500:
                # Server error (520, 502, etc.) - retry with exponential backoff
                print(f"      Server error {response.status_code}. Retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})", end="\r")
                time.sleep(delay)
                delay = min(delay * 2, 60)  # Exponential backoff, cap at 60s
                continue
            
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"      Request error: {str(e)[:50]}. Retrying in {delay:.1f}s... (attempt {attempt + 1}/{max_retries})", end="\r")
                time.sleep(delay)
                delay = min(delay * 2, 60)
            else:
                raise
    
    raise Exception(f"Failed to get embedding after {max_retries} attempts")


def embed_with_sentence_transformers(text: str, model) -> List[float]:
    """Generate embedding using sentence-transformers model."""
    return model.encode(text).tolist()


# ---------------------
# INGESTION
# ---------------------

print(f"✅ Ingesting into collection: {collection}")
print(f"📁 Reading PDFs from: {pdf_folder.resolve()}")
print(f"🔤 Embedding type: {EMBEDDING_TYPE}")
if CHUNK_SIZE and CHUNK_SIZE > 0:
    print(f"📏 Using chunk size: {CHUNK_SIZE} with overlap: {CHUNK_OVERLAP}")
else:
    print(f"📏 Chunking DISABLED - vectorizing at clause level (full combined clause text)")

files = [p for p in pdf_folder.iterdir() if p.suffix.lower() == ".pdf"]

if not files:
    print("⚠️  No PDF files found. Exiting.")
    sys.exit(0)

# Initialize embedding model based on type
embedding_model = None
vector_dim = VECTOR_DIMENSIONS[EMBEDDING_TYPE]

if EMBEDDING_TYPE == "sentence_transformers":
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    print(f"✅ Loaded sentence-transformers model (dimension: {vector_dim})")
elif EMBEDDING_TYPE == "openai":
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY is required for OpenAI embeddings")
        sys.exit(1)
    print(f"✅ Using OpenAI embeddings ({OPENAI_EMBEDDING_MODEL}, dimension: {vector_dim})")
else:
    print(f"❌ Unknown embedding type: {EMBEDDING_TYPE}")
    sys.exit(1)

# Connect to Qdrant - HARDCODED URL
client = QdrantClient(QDRANT_URL)
print(f"🔗 Connected to Qdrant at: {QDRANT_URL}")

# Create or recreate collection with cosine distance
try:
    # Check if collection exists
    client.get_collection(collection)
    # If exists, delete and recreate
    client.delete_collection(collection)
    print(f"🗑️  Deleted existing collection: {collection}")
except Exception:
    # Collection doesn't exist, which is fine
    pass

client.create_collection(
    collection_name=collection,
    vectors_config=VectorParams(size=vector_dim, distance="Cosine"),
)
print(f"📊 Collection configured for {vector_dim}-dimensional vectors")

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
        # This combines clauses by main clause ID (e.g., all sub-clauses of 7.1 are combined)
        combined_rows = combine_clauses_by_main(sections, clauses)
        
        print(f"   Found {len(sections)} sections, {len(clauses)} clauses, {len(combined_rows)} combined groups")
        
        # Create section map for efficient lookup
        section_map = {s["section_id"]: s for s in sections}
        
        # VECTORIZATION LEVEL EXPLANATION:
        # Currently vectorizing at the CHUNK level of COMBINED CLAUSE text:
        # 1. Clauses are combined by main clause ID (e.g., "7.1" + "7.1.1" + "7.1.2" = one combined row)
        # 2. Combined text is then chunked into 2000-char pieces (if chunking enabled)
        # 3. Each chunk gets its own vector embedding
        # 4. If chunking is disabled (CHUNK_SIZE=0), each combined clause gets ONE vector
        
        # Process each combined row
        for row in combined_rows:
            base_text = row.get("text", "").strip()
            if not base_text:
                continue
            
            # Chunk the text with character-based chunking and overlap
            # If CHUNK_SIZE is 0 or None, this returns the entire combined clause text as a single chunk
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
            
            # Create embeddings for each chunk (or single embedding if chunking disabled)
            for chunk_index, chunk in enumerate(char_chunks, start=1):
                # Generate embedding based on embedding type
                if EMBEDDING_TYPE == "openai":
                    # Add small delay to avoid rate limits
                    if chunk_index > 1:
                        time.sleep(0.1)  # 100ms delay between requests
                    vector = embed_with_openai(chunk)
                else:
                    vector = embed_with_sentence_transformers(chunk, embedding_model)
                
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
                    "chunk_part": chunk_index if len(char_chunks) > 1 else None,
                }
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload,
                )
                
                client.upsert(collection_name=collection, points=[point])
                total_chunks += 1
                
                if chunk_index % 10 == 0:
                    print(f"      Processed {chunk_index} chunks...", end="\r")
                
    except Exception as e:
        print(f"❌ Error processing {pdf_path.name}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n✅ Finished ingesting {len(files)} PDFs into '{collection}' collection ({total_chunks} chunks).")

