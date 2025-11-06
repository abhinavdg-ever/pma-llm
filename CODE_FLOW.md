# Sleep Coach LLM System - Code Flow Documentation

## Overview
The Sleep Coach LLM system is a layered architecture that processes sleep data, provides personalized insights, and answers questions using a local LLM (Llama) with vector database integration.

---

## 🔄 System Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application (app.py)              │
│  - REST API endpoints                                        │
│  - Request/Response handling                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              SleepCoachLLM (Main Orchestrator)              │
│  - Initializes all components                               │
│  - Coordinates query processing                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Analytics   │ │ Vector DB     │ │  Privacy     │
│  Database    │ │ (Qdrant)      │ │  Processor   │
└──────────────┘ └──────────────┘ └──────────────┘
```

---

## 📋 Component Initialization Flow

### 1. **Application Startup** (`app.py` → `startup_event()`)
```
1. Load environment variables (.env)
2. Validate configuration (Config.validate())
3. Initialize SleepCoachLLM() → Triggers component initialization
```

### 2. **SleepCoachLLM Initialization** (`sleep_coach_llm.py`)
```python
SleepCoachLLM.__init__()
├── PrivacyProcessor()          # Handles user ID pseudonymization
├── AnalyticsDatabase()          # MySQL connection for sleep data
├── CohortAnalytics()            # Cohort comparison metrics
├── VectorDatabase()            # Qdrant vector DB for knowledge base
├── SleepCoachAgent()           # Main query processing agent
└── ConversationLogger()         # Session logging
```

### 3. **Component Initialization Details**

#### **AnalyticsDatabase** (MySQL)
```
1. Connect to MySQL database (Config.DB_CONFIG)
2. Resolve table names (summary/details tables)
3. Get table columns dynamically
4. Store connection for query execution
```

#### **VectorDatabase** (Qdrant)
```
1. Initialize CustomEmbeddingClient (connects to embedding API)
2. Connect to Qdrant server (Config.QDRANT_URL)
3. Verify collection exists (Config.QDRANT_COLLECTION_NAME)
4. Fallback to mock if connection fails
```

#### **SleepCoachAgent**
```
1. Initialize LlamaClient (for LLM queries)
2. Initialize QueryClassifier (uses Llama)
3. Initialize SQLAgent (uses Llama + Analytics DB)
4. Set up tools (Analytics, Cohort, Knowledge, Chart)
```

---

## 🔍 Query Processing Flow

### Main Entry Point: `handle_user_query(user_id, query)`

```
┌─────────────────────────────────────────────────────────┐
│  User Query Received                                     │
│  "How was my sleep over the last 7 days?"               │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1: Privacy Processing                              │
│  - Convert user_id to pseudonymized ID (for logging)    │
│  - Keep original customer_id for DB queries            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: Agent Processing (process_query)               │
│  ┌───────────────────────────────────────────────────┐ │
│  │  Query Classifier                                  │ │
│  │  - Check if query is sleep-related                 │ │
│  │  - Classify into:                                  │ │
│  │    • Data Pull (SQL)                               │ │
│  │    • Knowledge                                     │ │
│  │    • LLM Core                                     │ │
│  │    • Off-Topic (routed to LLM Core)              │ │
│  └───────────────────────────────────────────────────┘ │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Data Pull    │ │  Knowledge   │ │  LLM Core    │
│ (SQL)        │ │              │ │              │
└──────────────┘ └──────────────┘ └──────────────┘
```

---

## 📊 Query Type Processing Details

### **Type 1: Data Pull (SQL) Query**
Example: *"How was my sleep over the last 7 days?"*

```
1. QueryClassifier.classify()
   └─> Returns: "Data Pull (SQL)"

2. SQLAgent.generate_sql(query, customer_id)
   ├─> Uses Llama to generate SQL query
   ├─> Provides database schema context
   └─> Returns SQL query

3. SQLAgent.run_query(sql_query)
   ├─> Execute SQL on MySQL database
   ├─> Serialize results (dates, decimals)
   └─> Return data rows

4. Format Response
   ├─> If ≤50 rows: Use raw data summary
   └─> If >50 rows: Use Llama for formatting

5. Return Response
   └─> Contains: sql_query, results, content, total_rows
```

### **Type 2: Knowledge Query**
Example: *"What's the importance of REM sleep?"*

```
1. QueryClassifier.classify()
   └─> Returns: "Knowledge"

2. VectorDatabase.similarity_search(query, k=3)
   ├─> CustomEmbeddingClient.embed_query(query)
   │   └─> POST to embedding API → Get 384-dim vector
   ├─> QdrantClient.search(query_vector, limit=k)
   └─> Return top-k similar documents

3. If Knowledge Found:
   ├─> Build context from retrieved documents
   ├─> LlamaClient.generate(prompt with context)
   └─> Append sources to response

4. If No Knowledge Found:
   └─> Fallback to LLM Core (general sleep advice)

5. Return Response
   └─> Contains: content, knowledge_sources, knowledge_used
```

### **Type 3: LLM Core Query**
Example: *"How can I improve my sleep?"*

```
1. QueryClassifier.classify()
   └─> Returns: "LLM Core"

2. Check if Off-Topic:
   ├─> If yes: Use special prompt (acknowledge + redirect)
   └─> If no: Use sleep advice prompt

3. LlamaClient.generate(prompt)
   └─> POST to Llama API → Stream response

4. Return Response
   └─> Contains: content (formatted sleep advice)
```

### **Type 4: Off-Topic Query**
Example: *"What is the capital of Delhi?"*

```
1. QueryClassifier.is_sleep_related(query)
   ├─> Check sleep keywords
   └─> Use Llama to verify → Returns False

2. QueryClassifier.classify()
   └─> Returns: "Off-Topic"

3. Route to LLM Core
   └─> Use special prompt that:
       - Acknowledges Sleep Coach specialization
       - Briefly answers if possible
       - Suggests sleep-related help

4. Return Response
   └─> Contains: content (polite redirect)
```

---

## 🔧 Component Details

### **CustomEmbeddingClient**
```python
embed_query(text: str) → List[float]
├─> POST to Config.EMBEDDING_API_URL
├─> Payload: {"text": text}
└─> Returns: 384-dimensional embedding vector

embed_documents(texts: List[str]) → List[List[float]]
└─> Calls embed_query() for each text
```

### **LlamaClient**
```python
generate(prompt: str, timeout: int) → str
├─> POST to Config.LLAMA_API_URL
├─> Payload: {"model": "llama3", "prompt": prompt, "stream": True}
├─> Stream response chunks
└─> Return complete response text
```

### **QueryClassifier**
```python
is_sleep_related(query: str) → bool
├─> Check sleep-related keywords
└─> Use Llama if no keywords found

classify(query: str) → str
├─> Check sleep relevance first
├─> Use Llama to classify into categories
└─> Returns: "Data Pull (SQL)" | "Knowledge" | "LLM Core" | "Off-Topic"
```

### **SQLAgent**
```python
generate_sql(query: str, customer_id: str) → str
├─> Get database schema (dynamic column fetching)
├─> Use Llama to generate SQL
├─> Clean SQL (remove markdown, validate)
└─> Auto-add GROUP BY if needed

run_query(sql_query: str) → List[Dict]
├─> Validate SELECT-only queries
├─> Execute on MySQL
├─> Serialize results (dates, decimals)
└─> Return rows

format_sql_response(query, sql_query, results) → str
└─> Use Llama to format results into natural language
```

### **VectorDatabase (Qdrant)**
```python
add_documents(documents: List[Dict])
├─> Extract texts and metadata
├─> Generate embeddings (CustomEmbeddingClient)
├─> Create PointStruct objects
└─> Upsert to Qdrant collection

similarity_search(query: str, k: int) → List[Dict]
├─> Generate query embedding
├─> Search Qdrant collection
└─> Format results with content, source, score
```

---

## 🔐 Privacy & Security Flow

### **PrivacyProcessor**
```
get_pseudo_id(user_id: str) → str
├─> Maps original user_id to pseudonymized ID
└─> Stores in memory mapping (user_{uuid})

redact_pii(data: Dict) → Dict
└─> Removes PII fields (name, email, phone, etc.)
```

### **Data Flow with Privacy**
```
User Query (original customer_id)
    ↓
PrivacyProcessor.get_pseudo_id() → pseudo_id (for logging)
    ↓
Database Query (uses original customer_id)
    ↓
Response (uses pseudo_id for logging)
```

---

## 📝 Logging Flow

### **ConversationLogger**
```
log_interaction(user_pseudo_id, query, response)
├─> Create session_id (date-based)
├─> Redact sensitive data from response
└─> Store in session history

get_session_history(user_pseudo_id) → List[Dict]
└─> Retrieve conversation history for user
```

---

## 🚀 API Endpoints Flow (FastAPI)

### **POST /query**
```
1. Receive QueryRequest {user_id, query}
2. Call sleep_coach.handle_user_query(user_id, query)
3. Return QueryResponse with:
   - response_type
   - content
   - sql_query (if applicable)
   - results (if applicable)
   - query_classification
   - knowledge_sources (if applicable)
```

### **POST /wearable**
```
1. Receive WearableDataRequest
2. Call sleep_coach.process_wearable_data(raw_data)
3. Store in AnalyticsDatabase
4. Return success status
```

### **POST /knowledge**
```
1. Receive KnowledgeDocumentRequest
2. Call sleep_coach.add_knowledge_documents(documents)
3. Embed documents and store in Qdrant
4. Return success status
```

---

## 🔄 Error Handling Flow

### **Connection Failures**
```
MySQL Connection Failed
└─> AnalyticsDatabase falls back to in-memory storage

Qdrant Connection Failed
└─> VectorDatabase falls back to mock (returns default knowledge)

Llama API Failed
└─> Returns error message in response.content

Embedding API Failed
└─> Raises exception, falls back to mock knowledge base
```

### **Query Processing Errors**
```
SQL Generation Error
└─> Returns user-friendly error message

SQL Execution Error
└─> Returns empty results, logs error

Vector Search Error
└─> Returns mock knowledge results

LLM Generation Error
└─> Returns error message in response
```

---

## 📦 Data Flow Summary

```
User Query
    ↓
[Privacy Processing]
    ↓
[Query Classification] → Llama API
    ↓
┌─────────────────────────────────────┐
│  Classification Branch               │
├─────────────────────────────────────┤
│  SQL Query:                          │
│    → SQL Generation (Llama)         │
│    → Database Query (MySQL)         │
│    → Format Results                  │
│                                      │
│  Knowledge Query:                   │
│    → Embed Query (Custom API)        │
│    → Vector Search (Qdrant)          │
│    → Generate Response (Llama)       │
│                                      │
│  LLM Core Query:                     │
│    → Generate Response (Llama)       │
└─────────────────────────────────────┘
    ↓
[Response Formatting]
    ↓
[Logging]
    ↓
Return to User
```

---

## 🛠️ Key Configuration Points

### **Environment Variables**
- `LLAMA_API_URL` - Required: Local Llama API endpoint
- `LLAMA_MODEL` - Llama model name (default: llama3)
- `QDRANT_URL` - Qdrant server URL (default: http://34.131.37.125:6333)
- `QDRANT_COLLECTION_NAME` - Collection name (default: docs)
- `EMBEDDING_API_URL` - Custom embedding service (default: http://34.131.37.125:8000/embed)
- `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE` - MySQL config

### **Database Tables**
- `ai_coach_modules_summary` - User profiles and risk scores
- `ai_coach_daily_sleep_details` - Daily sleep metrics

---

## 📈 Performance Considerations

1. **Embedding Generation**: Sequential API calls (can be parallelized)
2. **SQL Generation**: Uses Llama with timeout (30-45s)
3. **Vector Search**: Fast with Qdrant (milliseconds)
4. **Database Queries**: Depends on MySQL performance
5. **LLM Generation**: Streams response for better UX

---

## 🧪 Testing Flow

### **Demo Mode** (`main()` function)
```
1. Initialize SleepCoachLLM
2. Add knowledge documents
3. Process wearable data
4. Test different query types:
   - Personal data query
   - Cohort comparison
   - Knowledge query
```

---

This flow ensures proper separation of concerns, privacy handling, and scalable architecture for the Sleep Coach LLM system.

