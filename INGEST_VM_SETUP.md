# VM Setup Guide for PDF Ingestion

This guide explains how to run `ingest_pdfs_contracts_big.py` on your VM.

**Everything is hardcoded - just edit the API key and run!**

## Step 1: Transfer Files to VM

Transfer these files to your VM:
- `ingest_pdfs_contracts_big.py`
- `parse_contract.py`
- `contracts/` folder with PDF files

## Step 2: Install Dependencies

```bash
pip install requests qdrant-client pypdf
```

**Note:** OpenAI embeddings are hardcoded (no sentence-transformers needed).

## Step 3: Ready to Run!

**Everything is hardcoded - no editing needed!**
- ✅ OpenAI API key is hardcoded (line 43)
- ✅ Using OpenAI embeddings (`text-embedding-3-small`)
- ✅ Qdrant URL: `http://localhost:6333` (Qdrant on same VM)
- ✅ Chunking: **DISABLED** (vectorize at clause level)
- ✅ Collection: `contracts_big`

## Step 4: Prepare PDF Files

Ensure your PDF contract files are in a `contracts/` folder:

```bash
mkdir -p contracts
# Copy your PDF files to contracts/ folder
# Files should be named like:
# - Pacific_Coast_Longshore_Contract_Document_2022-2028.pdf
# - Pacific_Coast_Walking_Bosses_and_Foremens_Agreement_2022-2028.pdf
# - Pacific-Coast-Clerks-Contract-Document-2022-2028.pdf
```

## Step 5: Run the Script

```bash
python ingest_pdfs_contracts_big.py contracts_big
```

That's it! The script will:
1. Connect to Qdrant at `http://localhost:6333` (on same VM)
2. Recreate the `contracts_big` collection (WARNING: Deletes existing data!)
3. Process each PDF in `contracts/` folder
4. Extract text, parse clauses, combine by main clause ID
5. Generate OpenAI embeddings (1536-dimensional vectors)
6. Store in Qdrant (no chunking - each combined clause = one vector)

## Hardcoded Configuration

Everything is pre-configured in the script:

| Setting | Value | Location |
|---------|-------|----------|
| `EMBEDDING_TYPE` | `"openai"` | Line 42 |
| `OPENAI_API_KEY` | `(hardcoded)` | Line 43 |
| `OPENAI_EMBEDDING_MODEL` | `"text-embedding-3-small"` | Line 44 |
| `QDRANT_URL` | `"http://localhost:6333"` | Line 48 |
| `CHUNK_SIZE` | `0` (disabled) | Line 51 |

### Vectorization Level

**Chunking is DISABLED** (`CHUNK_SIZE=0`):
- Each combined clause (main + sub-clauses) gets ONE vector
- Vectorizing at clause level (semantically coherent)
- 1536-dimensional vectors from OpenAI

To change any setting, edit the constants at the top of the script (lines 37-52).

## Troubleshooting

### Error: "Folder contracts does not exist"
- Make sure you're running the script from the directory containing the `contracts/` folder
- Or create the folder and add PDF files

### Error: "OPENAI_API_KEY is required"
- Set `export OPENAI_API_KEY="your-key"`
- Or switch to `EMBEDDING_TYPE=sentence_transformers`

### Error: "Failed to connect to Qdrant"
- Verify Qdrant is running: `curl http://localhost:6333/health`
- Update `QDRANT_URL` if using a remote instance

### Slow Processing
- Sentence transformers: First run downloads the model (~5-10 min), subsequent runs are faster
- OpenAI: Rate limits may apply, script handles this automatically
- Large PDFs: Processing time depends on number of clauses and chunk size

## Vector Dimensions

- **Sentence Transformers**: 768 dimensions
- **OpenAI text-embedding-3-small**: 1536 dimensions
- **OpenAI text-embedding-3-large**: 3072 dimensions

The collection is automatically configured with the correct dimensions based on your embedding type.

## Output

The script will print progress:
```
✅ Ingesting into collection: contracts_big
📁 Reading PDFs from: /path/to/contracts
🔤 Embedding type: openai
📏 Chunking DISABLED - vectorizing at clause level
✅ Using OpenAI embeddings (text-embedding-3-small, dimension: 1536)
🔗 Connected to Qdrant at: http://localhost:6333
📊 Collection configured for 1536-dimensional vectors

📄 Processing Pacific_Coast_Longshore_Contract_Document_2022-2028.pdf...
   Found 50 sections, 234 clauses, 120 combined groups
      Processed 120 chunks...

✅ Finished ingesting 3 PDFs into 'contracts_big' collection (450 chunks).
```

