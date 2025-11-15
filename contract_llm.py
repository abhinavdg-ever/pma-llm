import logging
import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import mysql.connector
from tabulate import tabulate

SECTION_REF_REGEX = re.compile(r"(section\s+\d+[A-Za-z0-9\-\.]*)", re.IGNORECASE)
ARTICLE_REF_REGEX = re.compile(r"(article\s+\d+[A-Za-z0-9\-\.]*)", re.IGNORECASE)
APPENDIX_REF_REGEX = re.compile(r"(appendix\s+[A-Za-z0-9]+)", re.IGNORECASE)
CLAUSE_REF_REGEX = re.compile(r"(clause\s+\d+[A-Za-z0-9\-\.]*)", re.IGNORECASE)
# Core wage keywords - explicit wage/salary/pay terms that should trigger wage_schedule classification
CORE_WAGE_KEYWORDS = [
    "wage",
    "wages",
    "salary",
    "salaries",
    "hourly rate",
    "hourly rates",
    "pay rate",
    "pay rates",
    "rate of pay",
    "rates of pay",
    "hourly wage",
    "hourly wages",
    "hourly salary",
    "hourly salaries",
    "shift pay",
    "shift wages",
    "shift salary",
    "compensation",
    "payroll",
    "pay schedule",
    "wage schedule",
    "overtime",
]

# Secondary keywords that only indicate wage_schedule when combined with core wage keywords
SECONDARY_WAGE_KEYWORDS = [
    "plot",
    "tabulate",
    "fiscal year",
    "fy",
    "shift",
    "shifts",
]

WAGE_TABLE_NAME = "wage_schedule_pma"


class Config:
    """Configuration helpers for the contract insights engine."""

    QDRANT_URL = os.getenv("QDRANT_URL", "http://34.131.37.125:6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "contracts")
    QDRANT_COLLECTION_BIG = os.getenv("QDRANT_COLLECTION_NAME_BIG", "contracts_big")
    EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://34.131.37.125:8000/embed")
    EMBEDDING_DIMENSION = 768  # all-mpnet-base-v2 produces 768-dimensional vectors
    LLAMA_API_URL = os.getenv("LLAMA_API_URL", "http://34.131.37.125:11434/api/generate")
    LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3")
    LLAMA_TIMEOUT = int(os.getenv("LLAMA_TIMEOUT", "150"))
    LLAMA_TEMPERATURE = float(os.getenv("LLAMA_TEMPERATURE", "0.1"))  # Low temperature for more deterministic responses
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set via environment variable
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    @classmethod
    def validate(cls) -> None:
        if not cls.EMBEDDING_API_URL:
            raise ValueError("EMBEDDING_API_URL is not configured.")
        if not cls.QDRANT_URL:
            raise ValueError("QDRANT_URL is not configured.")


class CustomEmbeddingClient:
    """Thin wrapper over the embedding micro-service (sentence-transformers)."""

    def __init__(self, api_url: str):
        self.api_url = api_url

    def embed(self, text: str, timeout: int = 30) -> List[float]:
        response = requests.post(
            self.api_url,
            json={"text": text},
            headers={"content-type": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding")
        if not embedding:
            raise ValueError(f"Embedding API returned no vector: {data}")
        return embedding


class OpenAIEmbeddingClient:
    """OpenAI embedding client for generating embeddings."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/embeddings"
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")

    def embed(self, text: str, timeout: int = 60) -> List[float]:
        """Generate embedding using OpenAI API."""
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": text,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]


class OpenAIClient:
    """OpenAI client used to synthesize an answer from retrieved clauses."""

    def __init__(self, api_key: Optional[str], model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, timeout: Optional[int] = None, stream: bool = False) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        import logging
        logger = logging.getLogger("contract-insights-engine")
        logger.info(f"[OpenAI] Generating response with model: {self.model}, stream: {stream}, prompt_length: {len(prompt)}")

        effective_timeout = timeout or Config.LLAMA_TIMEOUT

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": Config.LLAMA_TEMPERATURE,
        }

        if stream:
            payload["stream"] = True
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=effective_timeout,
                stream=True,
            )
            response.raise_for_status()
            chunks = []
            for line in response.iter_lines():
                if not line:
                    continue
                line_text = line.decode("utf-8")
                if line_text.startswith("data: "):
                    data_str = line_text[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            chunks.append(delta["content"])
                    except json.JSONDecodeError:
                        continue
            return "".join(chunks).strip()

        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=effective_timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


class LlamaClient:
    """Optional LLM client used to synthesize an answer from retrieved clauses."""

    def __init__(self, api_url: Optional[str], model: str):
        self.api_url = api_url
        self.model = model

    def available(self) -> bool:
        return bool(self.api_url)

    def generate(self, prompt: str, timeout: Optional[int] = None, stream: bool = False) -> str:
        if not self.api_url:
            raise RuntimeError("LLAMA_API_URL is not configured.")

        import logging
        logger = logging.getLogger("contract-insights-engine")
        logger.info(f"[Llama] Generating response with model: {self.model}, stream: {stream}, prompt_length: {len(prompt)}")

        effective_timeout = timeout or Config.LLAMA_TIMEOUT

        if stream:
            with requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "temperature": Config.LLAMA_TEMPERATURE,
                },
                timeout=effective_timeout,
                stream=True,
            ) as response:
                response.raise_for_status()
                chunks = []
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        payload = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue
                    chunk = payload.get("response") or payload.get("data")
                    if chunk:
                        chunks.append(chunk)
                return "".join(chunks).strip()

        response = requests.post(
            self.api_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": Config.LLAMA_TEMPERATURE,
            },
            timeout=effective_timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()


class VectorDatabase:
    """Qdrant-backed vector search on the maritime contract corpus."""

    def __init__(self, qdrant_url: str, collection: str, embedder: CustomEmbeddingClient):
        self.collection = collection
        self.embedder = embedder
        self.client = None
        self.Filter = None
        self.FieldCondition = None
        self.MatchValue = None

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            import logging
            logger = logging.getLogger("contract-insights-engine")
            logger.info(f"[VectorDatabase] Attempting to connect to Qdrant at {qdrant_url}, collection: {collection}")
            
            self.client = QdrantClient(url=qdrant_url, timeout=10)
            self.client.get_collection(collection)  # Raises if missing.
            self.Filter = Filter
            self.FieldCondition = FieldCondition
            self.MatchValue = MatchValue
            logger.info(f"[VectorDatabase] Successfully connected to Qdrant collection: {collection}")
        except Exception as exc:  # pragma: no cover - graceful degradation.
            import logging
            logger = logging.getLogger("contract-insights-engine")
            logger.error(f"[VectorDatabase] FAILED to connect to Qdrant at {qdrant_url}, collection: {collection}. Error: {exc}")
            logger.warning("[VectorDatabase] Falling back to mock search - vector search will return mock data only")
            self.client = None
            self.Filter = None
            self.FieldCondition = None
            self.MatchValue = None

    def similarity_search(
        self,
        query: str,
        limit: int = 5,
        collection_override: Optional[str] = None,
        filter_conditions: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.client:
            return [
                {
                    "content": "Mock clause: Ensure safe working conditions per ILWU/PMA agreements.",
                    "source": "mock://contracts",
                    "score": 0.0,
                    "metadata": {},
                }
            ]

        query_vector = self.embedder.embed(query)
        search_filter = None
        if filter_conditions and self.Filter:
            try:
                search_filter = self.Filter(must=[cond for cond in filter_conditions if cond])
            except Exception:
                search_filter = None

        search_kwargs = {
            "collection_name": collection_override or self.collection,
            "query_vector": query_vector,
            "limit": limit,
        }

        if search_filter is not None:
            search_kwargs["query_filter"] = search_filter

        results = self.client.search(**search_kwargs)

        formatted: List[Dict[str, Any]] = []
        for point in results:
            payload = point.payload or {}
            formatted.append(
                {
                    "id": getattr(point, "id", None),
                    "score": point.score,
                    "content": payload.get("text") or payload.get("content", ""),
                    "source": payload.get("source", "unknown"),
                    "metadata": payload,
                }
            )
        return formatted

    def make_match_condition(self, key: str, value: Optional[str]):
        if value is None or not value.strip():
            return None
        if not self.FieldCondition or not self.MatchValue:
            return None
        try:
            return self.FieldCondition(key=key, match=self.MatchValue(value=value))
        except Exception:
            return None

    def exact_match_search(
        self,
        filter_conditions: List[Any],
        limit: int = 100,
        collection_override: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search by exact match on metadata fields (section_title, section_id, main_clause_id)."""
        if not self.client:
            return []

        search_filter = None
        if filter_conditions and self.Filter:
            try:
                search_filter = self.Filter(must=[cond for cond in filter_conditions if cond])
            except Exception:
                search_filter = None

        if not search_filter:
            return []

        # Use scroll to get all matching points (no vector search needed for exact match)
        try:
            results, _ = self.client.scroll(
                collection_name=collection_override or self.collection,
                scroll_filter=search_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            formatted: List[Dict[str, Any]] = []
            for point in results:
                payload = point.payload or {}
                formatted.append(
                    {
                        "id": getattr(point, "id", None),
                        "score": 1.0,  # Exact match gets perfect score
                        "content": payload.get("text") or payload.get("content", ""),
                        "source": payload.get("source", "unknown"),
                        "metadata": payload,
                    }
                )
            return formatted
        except Exception as exc:
            import logging
            logger = logging.getLogger("contract-insights-engine")
            logger.error(f"[VectorDatabase] Error in exact_match_search: {exc}")
            return []


class QueryClassifier:
    """Classify questions into contract knowledge, generic, or off-topic buckets."""

    def __init__(self, llama: LlamaClient):
        self.llama = llama

    def classify(self, query: str) -> str:
        query_lower = query.lower()

        # Check for explicit contract/document keywords that should take precedence
        contract_context_keywords = [
            "holiday", "holidays", "vacation", "vacations", "contract document", 
            "as per the contract", "per the contract", "according to contract",
            "meeting", "meetings", "benefit", "benefits", "guarantee", "guarantees",
            "summarize", "explain", "rules", "rule", "guidelines", "guideline",
            "policy", "policies", "jurisdiction"
        ]
        has_contract_context = any(keyword in query_lower for keyword in contract_context_keywords)
        
        # STRICT wage classification: Only classify as wage_schedule if explicit wage/salary keywords are present
        # Check for core wage keywords (wage, salary, hourly rate, etc.)
        has_core_wage_keyword = any(keyword in query_lower for keyword in CORE_WAGE_KEYWORDS)
        
        # If query has contract context keywords like "summarize", "explain", "rules" without wage keywords,
        # it should NOT be classified as wage_schedule
        if has_contract_context and not has_core_wage_keyword:
            # This is likely a contract knowledge query, not wage schedule
            pass  # Continue to check contract_knowledge keywords below
        elif has_core_wage_keyword:
            # Explicit wage/salary keyword found - classify as wage_schedule
            return "wage_schedule"
        
        # For "list" keyword, only classify as wage_schedule if combined with explicit wage keywords
        if "list" in query_lower:
            if has_core_wage_keyword:
                return "wage_schedule"
            # If "list" is combined with secondary keywords AND fiscal year/shift context, might be wage
            has_secondary_keyword = any(keyword in query_lower for keyword in SECONDARY_WAGE_KEYWORDS)
            if has_secondary_keyword and ("fiscal year" in query_lower or "fy" in query_lower or "shift" in query_lower):
                # Still require some wage context - check if it's clearly about wages
                if "rate" in query_lower or "pay" in query_lower:
                    return "wage_schedule"

        # Maritime/Contract keywords for contract_knowledge classification
        maritime_keywords = [
            "contract",
            "clause",
            "article",
            "pma",
            "ilwu",
            "walking boss",
            "walking bosses",
            "foremen",
            "foreman",
            "longshore",
            "longshoremen",
            "clerks",
            "clerk",
            "dispatcher",
            "agreement",
            "section",
            "grievance",
            "arbitration",
            "vacation",
            "vacations",
            "gang",
            "gangs",
            # Hours and Shifts
            "hours",
            "shift",
            "shifts",
            # Guarantees
            "guarantee",
            "guarantees",
            # Holidays
            "holiday",
            "holidays",
            # Scheduled Day Off
            "scheduled day off",
            "day off",
            # Dispatching, Registration, and Preference
            "dispatching",
            "registration",
            "preference",
            # Promotions and Training
            "promotion",
            "promotions",
            "training",
            # Organization of Gangs, Gang Sizes and Manning
            "manning",
            # No Strikes, Lockouts, and Work Stoppages
            "strike",
            "strikes",
            "lockout",
            "lockouts",
            "work stoppage",
            # Meetings for Registered Clerks
            "meeting",
            "meetings",
            "registered",
            # No Discrimination
            "discrimination",
            # Onerous Workload
            "workload",
            "onerous",
            # Efficient Operations
            "efficient",
            "operations",
            # Accident Prevention and Safety
            "accident",
            "safety",
            "prevention",
            # Joint Labor Relations Committees
            "labor relations",
            "committee",
            "committees",
            "procedures",
            # Good Faith Guarantee
            "good faith",
            # Union Security
            "union",
            "security",
            # Pay Guarantee Plan Rules
            "pay guarantee",
            "plan",
            "rules",
            "administration",
            # Lash Barge Jurisdiction
            "lash",
            "barge",
            "jurisdiction",
            # Term of Agreement
            "term",
            "review",
            # Welfare and Pension Plans
            "welfare",
            "pension",
            "plans",
            # Meal-related
            "meal",
            "meals",
            "meal time",
            "meal period",
        ]
        if any(token in query_lower for token in maritime_keywords):
            return "contract_knowledge"

        if not self.llama.available():
            print(f"[QueryClassifier] LLM API not available, falling back to generic_knowledge. API URL: {self.llama.api_url}")
            return "generic_knowledge"

        prompt = (
            "Classify the user question. Respond with ONLY one label:\n"
            "- contract_knowledge: ILWU/PMA maritime agreements, clauses, benefits, procedures.\n"
            "- generic_knowledge: General lifestyle, history, or unrelated but helpful topics.\n"
            "- off_topic: Neither maritime contracts nor helpful general context.\n\n"
            f"Question: {query}\nLabel:"
        )

        try:
            label = self.llama.generate(prompt, timeout=30).strip().lower()
            print(f"[QueryClassifier] LLM classification result: '{label}' for query: '{query[:50]}...'")
            if "contract" in label:
                return "contract_knowledge"
            if "off" in label:
                return "off_topic"
        except Exception as exc:  # pragma: no cover
            print(f"[QueryClassifier] Llama classification failed: {exc}. API URL: {self.llama.api_url}, Model: {self.llama.model}")

        return "generic_knowledge"


class ContractInsightsEngine:
    """Main orchestration class for contract Q&A."""

    def __init__(self):
        Config.validate()
        # Create embedders for both modes
        self.embedder_llama = CustomEmbeddingClient(Config.EMBEDDING_API_URL)  # sentence-transformers via API
        self.embedder_openai = OpenAIEmbeddingClient(Config.OPENAI_API_KEY) if Config.OPENAI_API_KEY else None  # OpenAI embeddings
        
        # Create vector DBs for both collections
        # Force "contracts" for Llama (not "docs" from env var)
        llama_collection = "contracts"
        self.vector_db_llama = VectorDatabase(Config.QDRANT_URL, llama_collection, self.embedder_llama)
        self.vector_db_openai = VectorDatabase(Config.QDRANT_URL, Config.QDRANT_COLLECTION_BIG, self.embedder_openai) if self.embedder_openai else None
        
        # Default to llama embedder for backward compatibility
        self.embedder = self.embedder_llama
        self.vector_db = self.vector_db_llama
        
        self.llama = LlamaClient(Config.LLAMA_API_URL, Config.LLAMA_MODEL)
        self.openai = OpenAIClient(Config.OPENAI_API_KEY, Config.OPENAI_MODEL)
        self.classifier = QueryClassifier(self.llama)
        self.logger = logging.getLogger("contract-insights-engine")
        self.wage_db_conn: Optional[mysql.connector.MySQLConnection] = None
        self.wage_schema: Optional[str] = None
        self.wage_metadata: Dict[str, List[str]] = {}
        self._initialize_wage_schedule()

    def _initialize_wage_schedule(self) -> None:
        host = os.getenv("WAGE_DB_HOST") or "72.60.96.212"
        user = os.getenv("WAGE_DB_USER") or "external"
        password = os.getenv("WAGE_DB_PASSWORD") or "External22x^^5420!"
        database = os.getenv("WAGE_DB_NAME") or "ym_raw_data_downstream"
        port = int(os.getenv("WAGE_DB_PORT", "3306"))

        try:
            conn = mysql.connector.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                autocommit=True,
            )
            cursor = conn.cursor()
            cursor.execute(f"DESCRIBE {WAGE_TABLE_NAME}")
            columns = cursor.fetchall()
            schema_lines = [f"- {col[0]} ({col[1]})" for col in columns]

            metadata: Dict[str, List[str]] = {}
            try:
                cursor.execute(f"SELECT DISTINCT EmployeeType FROM {WAGE_TABLE_NAME}")
                metadata["employee_types"] = sorted(row[0] for row in cursor.fetchall() if row[0])
                cursor.execute(f"SELECT DISTINCT SkillType FROM {WAGE_TABLE_NAME}")
                metadata["skill_types"] = sorted(row[0] for row in cursor.fetchall() if row[0])
                cursor.execute(f"SELECT DISTINCT ExperienceHours FROM {WAGE_TABLE_NAME}")
                metadata["experience_hours"] = sorted(row[0] for row in cursor.fetchall() if row[0])
            except Exception as meta_exc:  # pragma: no cover
                self.logger.warning("Failed to load wage metadata: %s", meta_exc)
            finally:
                cursor.close()

            self.wage_db_conn = conn
            self.wage_schema = "\n".join(schema_lines)
            self.wage_metadata = metadata
            self.logger.info(
                "Wage schedule connection established (table=%s, employee_types=%s)",
                WAGE_TABLE_NAME,
                metadata.get("employee_types", []),
            )
        except Exception as exc:
            self.logger.warning("Failed to initialize wage schedule connection: %s", exc)
            self.wage_db_conn = None
            self.wage_schema = None
            self.wage_metadata = {}

    def _get_llm_client(self, mode: str = "llama"):
        """Get the appropriate LLM client based on mode."""
        if mode == "openai":
            self.logger.info(f"[LLM Client] Using OpenAI API (model: {self.openai.model})")
            return self.openai
        self.logger.info(f"[LLM Client] Using Llama API (model: {self.llama.model}, url: {self.llama.api_url})")
        return self.llama

    def handle_query(self, query: str, top_k: int = 5, mode: str = "llama") -> Dict[str, Any]:
        # Use top_k=5 for both modes (OpenAI and Llama use same vector search)
        effective_top_k = top_k
        
        classification = self.classifier.classify(query)
        debug_info: Dict[str, Any] = {
            "query": query,
            "initial_classification": classification,
        }
        self.logger.info("Query received | classification=%s | mode=%s | top_k=%d | query=%s", classification, mode, effective_top_k, query)

        if classification == "contract_knowledge":
            # Check for exact match identifiers (section_title, section_id, main_clause_id)
            section_title = self._extract_section_title(query)
            section_id_exact = self._extract_section_id(query)
            main_clause_id = self._extract_main_clause_id(query)
            
            # Use exact match search if any identifier is found
            use_exact_match = bool(section_title or section_id_exact or main_clause_id)
            
            # Detect multiple employee types or use "all" if none specified
            doc_types = self._infer_doc_types(query)
            filter_conditions: List[Any] = []

            # If multiple types, we'll search across all of them (no doc_type filter)
            # If single type, filter by that type
            if len(doc_types) == 1:
                doc_type = doc_types[0]
                doc_type_condition = self.vector_db.make_match_condition("doc_type", doc_type)
                if doc_type_condition:
                    filter_conditions.append(doc_type_condition)
            else:
                # Multiple types or "all" - don't filter by doc_type, search all collections
                doc_type = None  # Will search all collections
            
            # Also check for section reference (existing logic)
            section_id_filter = self._extract_section_reference(query)
            if section_id_filter and not section_id_exact:
                # Section reference also triggers exact match
                use_exact_match = True
            
            # Validate: If using exact match (any identifier), require a specific contract to be specified
            if use_exact_match and doc_type is None:
                doc_type_label = "all"
                opening_text = "Document Not Specified"
                if debug_info is not None:
                    debug_info["error"] = "Document not specified for exact match search"
                    debug_info["note"] = "Exact match search requires a specific document type (longshore, clerks, or walking_bosses) to be specified"
                return {
                    "response_type": "contract_knowledge",
                    "content": "Document not specified. Please resubmit with the document type (longshore, clerks, or walking_bosses) specified in your query.",
                    "answer_points": [],
                    "tables": [],
                    "disclaimer": "To search for a specific section or clause, please include the document type in your query. For example: 'Section 2 in Longshore Agreement' or 'Clause 4.36 in Clerks Contract'.",
                    "sources": [],
                    "matches": [],
                    "total_matches": 0,
                    "opening": opening_text,
                    "doc_type": doc_type_label,
                    "query": query,
                }

            # Add exact match filters if found
            if section_title:
                title_condition = self.vector_db.make_match_condition("section_title", section_title)
                if title_condition:
                    filter_conditions.append(title_condition)
                debug_info["section_title"] = section_title
            
            if section_id_exact:
                section_condition = self.vector_db.make_match_condition("section_id", section_id_exact)
                if section_condition:
                    filter_conditions.append(section_condition)
                debug_info["section_id"] = section_id_exact
            
            if main_clause_id:
                clause_condition = self.vector_db.make_match_condition("main_clause_id", main_clause_id)
                if clause_condition:
                    filter_conditions.append(clause_condition)
                debug_info["main_clause_id"] = main_clause_id

            # Also check for section/clause reference (existing logic)
            section_id_filter = self._extract_section_reference(query)
            if section_id_filter and not section_id_exact:
                # Check if it's a clause reference (contains a dot, e.g., "2.1" or "4.36") or section reference (just number, e.g., "2")
                if '.' in section_id_filter:
                    # Clause reference - could be in main_clause_id or sub_clause_ids
                    # First, filter by section_id to get all chunks from that section
                    section_num = section_id_filter.split('.')[0]
                    section_condition = self.vector_db.make_match_condition("section_id", section_num)
                    if section_condition:
                        filter_conditions.append(section_condition)
                    # Store the clause reference for post-filtering (check sub_clause_ids)
                    use_exact_match = True
                    debug_info["clause_reference"] = section_id_filter  # Store for post-filtering
                    debug_info["section_id"] = section_num  # Also store section for filtering
                else:
                    # Section reference - use section_id filter
                    section_condition = self.vector_db.make_match_condition("section_id", section_id_filter)
                    if section_condition:
                        filter_conditions.append(section_condition)
                    use_exact_match = True
                    debug_info["section_id"] = section_id_filter

            self.logger.info(
                "Contract query routed | doc_types=%s | section_id=%s | section_title=%s | main_clause_id=%s | use_exact_match=%s",
                doc_types,
                section_id_exact or section_id_filter or "none",
                section_title or "none",
                main_clause_id or "none",
                use_exact_match,
            )
            debug_info["doc_types"] = doc_types
            debug_info["doc_type"] = doc_types[0] if len(doc_types) == 1 else "all"
            debug_info["use_exact_match"] = use_exact_match

            # Use contracts_big collection + OpenAI embeddings for OpenAI mode
            # Use contracts collection + sentence-transformers embeddings for Llama mode
            if mode == "openai":
                if not self.embedder_openai or not self.vector_db_openai:
                    error_msg = (
                        "OpenAI embeddings not configured. "
                        "Please set OPENAI_API_KEY environment variable to use OpenAI mode."
                    )
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                collection_name = Config.QDRANT_COLLECTION_BIG
                vector_db = self.vector_db_openai
                self.logger.info(f"[Vector Search] Using OpenAI embeddings + collection: {collection_name}")
            else:
                # Force "contracts" collection for Llama mode (not "docs")
                collection_name = "contracts"
                vector_db = self.vector_db_llama
                self.logger.info(f"[Vector Search] Using sentence-transformers embeddings + collection: {collection_name}")
            
            # Use exact match search if identifiers found, otherwise use similarity search
            if use_exact_match:
                payload = self._answer_with_exact_match(
                    query,
                    filter_conditions=filter_conditions or None,
                    collection_name=collection_name,
                    debug_info=debug_info,
                    mode=mode,
                    vector_db=vector_db,
                    doc_type=doc_types[0] if len(doc_types) == 1 else None,
                    doc_types=doc_types if len(doc_types) > 1 else None,
                )
            else:
                payload = self._answer_with_contract_knowledge(
                    query,
                    effective_top_k,
                    doc_type=doc_types[0] if len(doc_types) == 1 else None,
                    doc_types=doc_types if len(doc_types) > 1 else None,
                    collection_name=collection_name,
                    filter_conditions=filter_conditions or None,
                    debug_info=debug_info,
                    mode=mode,
                    vector_db=vector_db,
                )
        elif classification == "wage_schedule":
            debug_info["doc_type"] = "wage_schedule"
            payload = self._answer_wage_schedule(query, debug_info, mode=mode)
        elif classification == "generic_knowledge":
            payload = self._answer_with_generic_llm(query, mode=mode)
            debug_info["doc_type"] = "n/a"
        else:
            payload = self._answer_off_topic(query, mode=mode)
            debug_info["doc_type"] = "n/a"

        payload.setdefault("answer_points", [])
        payload.setdefault("disclaimer", None)
        payload.setdefault("sources", [])
        payload.setdefault("matches", [])
        payload.setdefault("total_matches", len(payload.get("matches", [])))
        payload.setdefault("content", "")
        payload["query"] = query
        label_map = {
            "contract_knowledge": "Contract Knowledge",
            "generic_knowledge": "General Knowledge",
            "off_topic": "Off-Topic",
            "wage_schedule": "Wage Schedule · SQL",
        }
        if classification == "contract_knowledge":
            doc_type = payload.get("doc_type") or self._infer_doc_type(query)
            type_label_map = {
                "longshore": "Longshore",
                "clerks": "Clerks",
                "walking_bosses": "Walking Bosses",
            }
            doc_label = type_label_map.get(doc_type, doc_type.title())
            payload["query_classification"] = f"Contract Knowledge · {doc_label}"
            payload["doc_type"] = doc_type
            debug_info["doc_type"] = doc_type
        elif classification == "wage_schedule":
            payload["query_classification"] = "Wage Schedule · SQL"
        else:
            payload["query_classification"] = label_map.get(classification, classification)
        payload["debug"] = debug_info
        return payload

    def _enhance_query_for_specific_topics(self, query: str) -> str:
        """Enhance query with topic-specific terms to improve vector search relevance."""
        query_lower = query.lower()
        
        # Enhance holiday queries - add specific terms that differentiate from vacations
        if "holiday" in query_lower:
            enhanced = f"{query} paid holidays holiday schedule New Year Christmas Thanksgiving Memorial Day Labor Day Independence Day"
            self.logger.info(f"Enhanced holiday query: {enhanced}")
            return enhanced
        
        # Enhance vacation queries - add specific terms
        if "vacation" in query_lower and "holiday" not in query_lower:
            enhanced = f"{query} vacation pay qualifying hours vacation weeks earned vacation"
            self.logger.info(f"Enhanced vacation query: {enhanced}")
            return enhanced
        
        return query
    
    def _filter_matches_by_topic(self, matches: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Filter out irrelevant matches based on query topic."""
        query_lower = query.lower()
        
        # If query is about holidays, filter out vacation-related content
        if "holiday" in query_lower and "vacation" not in query_lower:
            holiday_keywords = ["holiday", "paid holiday", "holiday schedule", "new year", "christmas", 
                              "thanksgiving", "memorial day", "labor day", "independence day", "columbus day",
                              "veterans day", "presidents day", "martin luther king"]
            vacation_keywords = ["vacation pay", "qualifying hours", "vacation week", "earned vacation",
                               "vacation with pay", "basic vacation", "additional vacation"]
            
            filtered = []
            holiday_matches = []
            for match in matches:
                content_lower = match.get("content", "").lower()
                # Check if content is clearly vacation-related
                has_vacation_context = any(kw in content_lower for kw in vacation_keywords)
                has_holiday_context = any(kw in content_lower for kw in holiday_keywords)
                
                # Prioritize holiday matches
                if has_holiday_context:
                    holiday_matches.append(match)
                
                # Exclude if it's vacation-related and has no holiday context
                if has_vacation_context and not has_holiday_context:
                    self.logger.info(f"Filtered out vacation-related match for holiday query: {content_lower[:100]}...")
                    continue
                filtered.append(match)
            
            # If we have holiday matches, prioritize them
            if holiday_matches:
                # Keep holiday matches first, then other filtered matches
                holiday_match_ids = {id(m) for m in holiday_matches}
                other_filtered = [m for m in filtered if id(m) not in holiday_match_ids]
                filtered = holiday_matches + other_filtered
                self.logger.info(f"Filtered {len(matches)} matches to {len(filtered)} holiday-relevant matches ({len(holiday_matches)} holiday-specific)")
                return filtered
            elif filtered:
                self.logger.info(f"Filtered {len(matches)} matches to {len(filtered)} (no holiday-specific matches found, but kept non-vacation matches)")
                return filtered
        
        # If query is about vacations, filter out holiday-related content
        if "vacation" in query_lower and "holiday" not in query_lower:
            holiday_keywords = ["paid holiday", "holiday schedule", "new year", "christmas", "thanksgiving"]
            vacation_keywords = ["vacation pay", "qualifying hours", "vacation week", "earned vacation",
                               "vacation with pay", "basic vacation", "additional vacation"]
            
            filtered = []
            for match in matches:
                content_lower = match.get("content", "").lower()
                # Check if content is clearly holiday-related
                has_holiday_context = any(kw in content_lower for kw in holiday_keywords)
                has_vacation_context = any(kw in content_lower for kw in vacation_keywords)
                
                # Exclude if it's holiday-related and has no vacation context
                if has_holiday_context and not has_vacation_context:
                    self.logger.info(f"Filtered out holiday-related match for vacation query: {content_lower[:100]}...")
                    continue
                filtered.append(match)
            
            if filtered:
                self.logger.info(f"Filtered {len(matches)} matches to {len(filtered)} vacation-relevant matches")
                return filtered
        
        return matches

    def _answer_with_contract_knowledge(
        self,
        query: str,
        top_k: int,
        doc_type: Optional[str] = None,
        doc_types: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
        filter_conditions: Optional[List[Any]] = None,
        debug_info: Optional[Dict[str, Any]] = None,
        mode: str = "llama",
        vector_db: Optional[VectorDatabase] = None,
    ) -> Dict[str, Any]:
        # Use provided vector_db or fall back to default based on mode
        if vector_db is None:
            vector_db = self.vector_db_llama if mode == "llama" else (self.vector_db_openai or self.vector_db_llama)
        
        # Enhance query for better vector search results
        enhanced_query = self._enhance_query_for_specific_topics(query)
        
        # If multiple doc_types, search across all of them
        if doc_types and len(doc_types) > 1:
            all_matches = []
            # Search each document type collection separately
            for dt in doc_types:
                dt_filter = [vector_db.make_match_condition("doc_type", dt)]
                if filter_conditions:
                    dt_filter.extend(filter_conditions)
                # Search with limit per type, then combine and re-rank
                type_matches = vector_db.similarity_search(
                    enhanced_query,
                    limit=top_k * 2,  # Get more per type to allow for re-ranking
                    collection_override=collection_name,
                    filter_conditions=dt_filter,
                )
                all_matches.extend(type_matches)
            # Sort by score and take top_k
            all_matches.sort(key=lambda x: x.get("score", 0), reverse=True)
            matches = all_matches[:top_k * 2]  # Get more before filtering
            # Filter matches by topic relevance
            matches = self._filter_matches_by_topic(matches, query)
            matches = matches[:top_k]
        else:
            # Single doc_type or none - normal search
            matches = vector_db.similarity_search(
                enhanced_query,
                limit=top_k * 2,  # Get more before filtering
                collection_override=collection_name,
                filter_conditions=filter_conditions,
            )
            # Filter matches by topic relevance
            matches = self._filter_matches_by_topic(matches, query)
            matches = matches[:top_k]
        if not matches:
            # Handle doc_type for no matches case
            if doc_types and len(doc_types) > 1:
                doc_type_label = "all"
                doc_labels = []
                type_label_map = {
                    "walking_bosses": "Walking Bosses and Foremen",
                    "clerks": "Clerks",
                    "longshore": "Longshore",
                }
                for dt in doc_types:
                    doc_labels.append(type_label_map.get(dt, dt.title()))
                # Extract regex operation outside f-string (cannot use backslashes in f-string expressions)
                cleaned_query = re.sub(r'\s+', ' ', query.strip())
                opening_text = f"Key findings from the {', '.join(doc_labels)} contracts regarding \"{cleaned_query}\":"
            else:
                doc_type_label = doc_type or (debug_info.get("doc_type") if debug_info else "longshore")
                opening_text = self._craft_opening_text(query, doc_type_label)
            
            if debug_info is not None:
                debug_info["retrieved_sources"] = []
                debug_info["note"] = "No matches from vector search"
            return {
                "response_type": "contract_knowledge",
                "content": "No relevant contract passages were located in the indexed agreements.",
                "answer_points": [],
                "tables": [],
                "disclaimer": "Please refer to exact contract text for detailed understanding.",
                "sources": [],
                "matches": [],
                "total_matches": 0,
                "opening": opening_text,
                "doc_type": doc_type_label,
            }

        sources = self._build_source_entries(matches)
        answer_points, disclaimer, tables = self._synthesize_contract_answer(query, matches, sources, mode=mode)
        content = self._assemble_contract_content(answer_points, disclaimer, sources)

        if debug_info is not None:
            # Use top_k sources for debug info
            sources_limit = top_k
            debug_info["retrieved_sources"] = [
                {
                    "source": src.get("source"),
                    "section_title": src.get("section_heading"),
                    "clause": src.get("clause"),
                    "main_clause_id": src.get("main_clause_id"),
                    "sub_clause_ids": src.get("sub_clause_ids"),
                    "page": src.get("page"),
                    "score": src.get("score"),
                }
                for src in sources[:sources_limit]
            ]
            self.logger.info(
                "Vector matches | count=%d | details=%s",
                len(debug_info["retrieved_sources"]),
                debug_info["retrieved_sources"],
            )

        # Handle opening text for multiple doc types
        if doc_types and len(doc_types) > 1:
            doc_labels = []
            type_label_map = {
                "walking_bosses": "Walking Bosses and Foremen",
                "clerks": "Clerks",
                "longshore": "Longshore",
            }
            for dt in doc_types:
                doc_labels.append(type_label_map.get(dt, dt.title()))
            # Extract regex operation outside f-string (cannot use backslashes in f-string expressions)
            cleaned_query = re.sub(r'\s+', ' ', query.strip())
            opening_text = f"Key findings from the {', '.join(doc_labels)} contracts regarding \"{cleaned_query}\":"
            doc_type_for_opening = "all"  # Multiple types
        else:
            doc_type_for_opening = doc_type or (debug_info.get("doc_type") if debug_info else "longshore")
            opening_text = self._craft_opening_text(query, doc_type_for_opening)

        return {
            "response_type": "contract_knowledge",
            "content": content,
            "answer_points": answer_points,
            "disclaimer": disclaimer,
            "sources": sources[:5],  # Both modes use top 5 sources
            "tables": tables,  # Add tables to response
            "matches": matches,
            "total_matches": len(matches),
            "opening": opening_text,
            "doc_type": doc_type_for_opening,
        }

    def _answer_with_exact_match(
        self,
        query: str,
        filter_conditions: Optional[List[Any]] = None,
        collection_name: Optional[str] = None,
        debug_info: Optional[Dict[str, Any]] = None,
        mode: str = "llama",
        vector_db: Optional[VectorDatabase] = None,
        doc_type: Optional[str] = None,
        doc_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Handle queries with exact match on section_title, section_id, or main_clause_id.
        Returns all matching chunks and generates a summary."""
        # Use provided vector_db or fall back to default based on mode
        if vector_db is None:
            vector_db = self.vector_db_llama if mode == "llama" else (self.vector_db_openai or self.vector_db_llama)
        
        # Use exact match search to get all chunks
        matches = vector_db.exact_match_search(
            filter_conditions=filter_conditions or [],
            limit=1000,  # Get all matching chunks
            collection_override=collection_name,
        )
        
        # Post-filter: If clause_reference was specified, filter by main_clause_id or sub_clause_ids
        clause_reference = debug_info.get("clause_reference") if debug_info else None
        self.logger.info(
            "Exact match search | collection=%s | matches=%d | clause_reference=%s",
            collection_name,
            len(matches),
            clause_reference or "none"
        )
        if clause_reference and matches:
            filtered_matches = []
            for match in matches:
                metadata = match.get("metadata", {})
                main_clause_id = str(metadata.get("main_clause_id", "")).strip()
                sub_clause_ids = str(metadata.get("sub_clause_ids", "")).strip()
                
                # Check if clause_reference matches main_clause_id
                if main_clause_id == clause_reference:
                    filtered_matches.append(match)
                    self.logger.debug("Match found via main_clause_id: %s", main_clause_id)
                    continue
                
                # Check if clause_reference is in sub_clause_ids (comma-separated list)
                if sub_clause_ids:
                    sub_clauses = [s.strip() for s in sub_clause_ids.split(',')]
                    # Check exact match
                    if clause_reference in sub_clauses:
                        filtered_matches.append(match)
                        self.logger.debug("Match found via sub_clause_ids (exact): %s in [%s]", clause_reference, sub_clause_ids)
                        continue
                    # Also check if any sub-clause starts with the clause_reference (e.g., "4.36" matches "4.36.1")
                    for sub_clause in sub_clauses:
                        if sub_clause.startswith(clause_reference + '.'):
                            filtered_matches.append(match)
                            self.logger.debug("Match found via sub_clause_ids (prefix): %s matches %s", clause_reference, sub_clause)
                            break
                    if filtered_matches and filtered_matches[-1] == match:
                        continue
                
                # Log for debugging
                self.logger.debug(
                    "No match: clause_reference=%s, main_clause_id=%s, sub_clause_ids=%s",
                    clause_reference,
                    main_clause_id,
                    sub_clause_ids
                )
            
            if filtered_matches:
                total_before = len(matches)
                matches = filtered_matches
                self.logger.info(
                    "Post-filtered to %d matches for clause %s (out of %d total matches from section)",
                    len(matches),
                    clause_reference,
                    total_before
                )
            else:
                # No matches found for the specific clause
                matches = []
        
        if not matches:
            # If no exact matches found and section/clause was mentioned, return error message
            # Don't use similarity search - only use exact match for section/clause queries
            section_ref = self._extract_section_reference(query)
            section_title = self._extract_section_title(query)
            section_id_exact = self._extract_section_id(query)
            main_clause_id = self._extract_main_clause_id(query)
            
            # If any section/clause identifier was mentioned, we should have found matches
            # If not, return error message (don't fall back to similarity search)
            if section_ref or section_title or section_id_exact or main_clause_id:
                # Build the reference string for the error message
                ref_parts = []
                if section_ref:
                    ref_parts.append(f"section/clause '{section_ref}'")
                if section_title:
                    ref_parts.append(f"section title '{section_title}'")
                if section_id_exact:
                    ref_parts.append(f"section_id '{section_id_exact}'")
                if main_clause_id:
                    ref_parts.append(f"main_clause_id '{main_clause_id}'")
                
                ref_string = ", ".join(ref_parts)
                
                self.logger.warning(
                    "No exact matches found for %s in collection %s, returning error message",
                    ref_string,
                    collection_name
                )
                
                # Return error message
                if doc_types and len(doc_types) > 1:
                    doc_type_label = "all"
                    doc_labels = []
                    type_label_map = {
                        "walking_bosses": "Walking Bosses and Foremen",
                        "clerks": "Clerks",
                        "longshore": "Longshore",
                    }
                    for dt in doc_types:
                        doc_labels.append(type_label_map.get(dt, dt.title()))
                    cleaned_query = re.sub(r'\s+', ' ', query.strip())
                    opening_text = f"Key findings from the {', '.join(doc_labels)} contracts regarding \"{cleaned_query}\":"
                else:
                    doc_type_label = doc_type or (debug_info.get("doc_type") if debug_info else "longshore")
                    opening_text = self._craft_opening_text(query, doc_type_label)
                
                if debug_info is not None:
                    debug_info["retrieved_sources"] = []
                    debug_info["note"] = f"Can't find {ref_string}"
                    debug_info["collection_used"] = collection_name
                
                # Create a more helpful error message
                error_content = f"Can't find this section/clause: {ref_string}."
                if section_ref and '.' in section_ref:
                    error_content += f" Please verify the Section or Clause ID. The clause '{section_ref}' was not found in the {doc_type_label} contract document."
                else:
                    error_content += " Please verify the section_id, section_title, main_clause_id, or section/clause reference you provided."
                
                return {
                    "response_type": "contract_knowledge",
                    "content": error_content,
                    "answer_points": [],
                    "tables": [],
                    "disclaimer": "The specified section or clause was not found in the contract documents. Please check the section/clause identifier and try again.",
                    "sources": [],
                    "matches": [],
                    "total_matches": 0,
                    "opening": opening_text,
                    "doc_type": doc_type_label,
                    "query": query,
                }
            
            # If no section/clause was mentioned and no matches, return standard no matches message
            if doc_types and len(doc_types) > 1:
                doc_type_label = "all"
                doc_labels = []
                type_label_map = {
                    "walking_bosses": "Walking Bosses and Foremen",
                    "clerks": "Clerks",
                    "longshore": "Longshore",
                }
                for dt in doc_types:
                    doc_labels.append(type_label_map.get(dt, dt.title()))
                cleaned_query = re.sub(r'\s+', ' ', query.strip())
                opening_text = f"Key findings from the {', '.join(doc_labels)} contracts regarding \"{cleaned_query}\":"
            else:
                doc_type_label = doc_type or (debug_info.get("doc_type") if debug_info else "longshore")
                opening_text = self._craft_opening_text(query, doc_type_label)
            
            if debug_info is not None:
                debug_info["retrieved_sources"] = []
                debug_info["note"] = "No matches from exact match search"
            return {
                "response_type": "contract_knowledge",
                "content": "No matching chunks were found for the specified identifier(s).",
                "answer_points": [],
                "tables": [],
                "disclaimer": "Please verify the section_title, section_id, or main_clause_id you provided.",
                "sources": [],
                "matches": [],
                "total_matches": 0,
                "opening": opening_text,
                "doc_type": doc_type_label,
                "query": query,
            }
        

        sources = self._build_source_entries(matches)
        
        # Generate summary using LLM
        summary = self._generate_summary_for_chunks(query, matches, sources, mode=mode)
        
        # Parse summary into bullet points (split by newlines or bullet markers)
        answer_points = []
        if summary:
            # Split by newlines and filter out empty lines
            lines = [line.strip() for line in summary.split('\n') if line.strip()]
            for line in lines:
                # Remove bullet markers if present (•, -, *, etc.)
                cleaned_line = re.sub(r'^[\s•\-\*\+]\s*', '', line)
                if cleaned_line:
                    answer_points.append(cleaned_line)
        
        # If no bullet points were extracted, use the summary as a single point
        if not answer_points and summary:
            answer_points = [summary]
        
        # Create opening text
        if doc_types and len(doc_types) > 1:
            doc_labels = []
            type_label_map = {
                "walking_bosses": "Walking Bosses and Foremen",
                "clerks": "Clerks",
                "longshore": "Longshore",
            }
            for dt in doc_types:
                doc_labels.append(type_label_map.get(dt, dt.title()))
            cleaned_query = re.sub(r'\s+', ' ', query.strip())
            opening_text = f"Summary of chunks from the {', '.join(doc_labels)} contracts matching the specified identifier(s):"
        else:
            doc_type_for_opening = doc_type or (debug_info.get("doc_type") if debug_info else "longshore")
            opening_text = f"Summary of chunks from the {doc_type_for_opening} contract matching the specified identifier(s):"
        
        # Assemble content using answer_points (bullet points)
        if answer_points:
            content = self._assemble_contract_content(
                answer_points,
                "This is a summary of all chunks matching the specified identifier(s). Please refer to exact contract text for detailed understanding.",
                sources
            )
        else:
            content = "Retrieved chunks for the specified identifier(s)."
        
        if debug_info is not None:
            debug_info["retrieved_sources"] = [
                {
                    "source": src.get("source"),
                    "section_title": src.get("section_heading"),
                    "clause": src.get("clause"),
                    "main_clause_id": src.get("main_clause_id"),
                    "sub_clause_ids": src.get("sub_clause_ids"),
                    "page": src.get("page"),
                    "score": src.get("score"),
                }
                for src in sources[:20]  # Show up to 20 sources in debug
            ]
            debug_info["total_chunks"] = len(matches)
            self.logger.info(
                "Exact match search | chunks=%d | details=%s",
                len(matches),
                debug_info["retrieved_sources"][:5] if debug_info["retrieved_sources"] else [],
            )

        # Extract disclaimer from content if it was assembled, otherwise use default
        default_disclaimer = "This is a summary of all chunks matching the specified identifier(s). Please refer to exact contract text for detailed understanding."
        disclaimer = default_disclaimer
        
        return {
            "response_type": "contract_knowledge",
            "content": content,
            "answer_points": answer_points,
            "disclaimer": disclaimer,
            "sources": sources[:10],  # Show top 10 sources
            "tables": [],
            "matches": matches,
            "total_matches": len(matches),
            "opening": opening_text,
            "doc_type": doc_type_for_opening if not (doc_types and len(doc_types) > 1) else "all",
            "query": query,
        }

    def _generate_summary_for_chunks(
        self,
        query: str,
        matches: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
        mode: str = "llama",
    ) -> str:
        """Generate a summary of the retrieved chunks using LLM."""
        if not matches or not sources:
            return ""
        
        llm_client = self._get_llm_client(mode)
        if not llm_client.available():
            # Fallback: return a simple summary
            return f"Found {len(matches)} chunk(s) matching the specified identifier(s)."
        
        # Combine all chunk contents
        chunk_texts = []
        for i, src in enumerate(sources[:50], 1):  # Limit to 50 chunks for summary
            excerpt = src.get("excerpt", "").strip()
            if excerpt:
                section = src.get("section_heading", "Unknown Section")
                clause = src.get("clause", "Unknown Clause")
                chunk_texts.append(f"Chunk {i} (Section: {section}, Clause: {clause}):\n{excerpt}\n")
        
        combined_text = "\n".join(chunk_texts)
        
        # Truncate if too long (keep last 8000 chars to preserve context)
        if len(combined_text) > 10000:
            combined_text = "..." + combined_text[-8000:]
        
        prompt = f"""You are analyzing contract document chunks retrieved by exact match on section_title, section_id, or main_clause_id.

The user query was: "{query}"

Below are {len(chunk_texts)} chunks that match the specified identifier(s):

{combined_text}

Please provide a comprehensive summary of these chunks. The summary should:
1. Identify the main topics and themes covered
2. Highlight key provisions, requirements, or rules
3. Note any important details or exceptions
4. Format as bullet points (one point per line, each point should be a complete sentence)
5. Aim for 5-10 bullet points that cover the key information

Format your response as bullet points, one per line. Each bullet point should be a complete, standalone sentence.

Summary:"""

        try:
            summary = llm_client.generate(prompt, timeout=60)
            return summary.strip()
        except Exception as exc:
            self.logger.error(f"Error generating summary: {exc}")
            return f"Found {len(matches)} chunk(s) matching the specified identifier(s). Unable to generate summary due to error."

    def _build_source_entries(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for match in matches:
            metadata = match.get("metadata") or {}
            start_page = metadata.get("start_page")
            end_page = metadata.get("end_page")
            section_title = metadata.get("section_title") or metadata.get("section_heading")
            # Handle new format: main_clause_id and sub_clause_ids, or fallback to old format
            main_clause_id = metadata.get("main_clause_id")
            sub_clause_ids = metadata.get("sub_clause_ids")
            clause_id = metadata.get("clause_id") or metadata.get("clause")
            # Use main_clause_id if available, otherwise use clause_id
            display_clause_id = main_clause_id or clause_id
            page_display: Optional[int] = None
            if isinstance(start_page, int) and isinstance(end_page, int):
                page_display = start_page if start_page == end_page else start_page
            elif isinstance(start_page, int):
                page_display = start_page
            elif isinstance(start_page, str) and start_page.isdigit():
                page_display = int(start_page)
            entries.append(
                {
                    "source": match.get("source", "unknown"),
                    "page": page_display,
                    "score": match.get("score"),
                    "excerpt": match.get("content", "").strip(),
                    "section_heading": section_title,
                    "clause": display_clause_id,
                    "main_clause_id": main_clause_id,
                    "sub_clause_ids": sub_clause_ids,
                    "section_id": metadata.get("section_id"),
                    "start_page": start_page,
                    "end_page": end_page,
                }
            )
        return entries

    def _synthesize_contract_answer(
        self,
        query: str,
        matches: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
        mode: str = "llama",
    ) -> Tuple[List[str], str, List[Dict[str, Any]]]:
        default_disclaimer = "Please refer to exact contract text for detailed understanding."
        answer_points: List[str] = []
        tables: List[Dict[str, Any]] = []
        disclaimer = self._normalize_disclaimer(default_disclaimer)

        llm_client = self._get_llm_client(mode)
        api_name = "OpenAI" if mode == "openai" else "Llama"
        self.logger.info(f"[Contract Synthesis] Using {api_name} API to synthesize answer from {len(sources)} sources")
        if llm_client.available():
            # Pre-clean excerpts before sending to LLM
            cleaned_sources = []
            for src in sources:
                excerpt = src.get('excerpt', '')[:800]
                cleaned_excerpt = self._clean_excerpt_for_llm(excerpt)
                if cleaned_excerpt:
                    cleaned_sources.append(f"Source: {self._format_source_label(src)}\nClause: {cleaned_excerpt}")
            context_snippets = "\n\n".join(cleaned_sources)
            prompt = (
                "You are a contract analyst for ILWU/PMA maritime agreements. "
                "Your task is to provide clear, concise summaries that directly answer the user's question.\n\n"
                "IMPORTANT: You MUST return valid JSON only. Do not include any text before or after the JSON.\n\n"
                "JSON schema:\n"
                "{\n"
                '  "answer": ["First clear bullet point", "Second clear bullet point", "..."],\n'
                '  "tables": [{"headers": ["Column1", "Column2"], "rows": [["Value1", "Value2"], ["Value3", "Value4"]]}],  // OPTIONAL: Use for structured data like holiday lists, benefit schedules, etc.\n'
                '  "disclaimer": "Please refer to exact contract text for detailed understanding."\n'
                "}\n\n"
                "TABULATION RULES:\n"
                "- If the question asks for lists, schedules, or tabular data (e.g., 'holiday list', 'vacation schedule', 'benefits'), use the 'tables' field\n"
                "- Each table should have 'headers' (array of column names) and 'rows' (array of arrays, where each inner array is a row)\n"
                "- Example for holidays: {\"headers\": [\"Holiday\", \"Date/Details\"], \"rows\": [[\"New Year's Day\", \"January 1\"], [\"Martin Luther King's Day\", \"Third Monday in January\"]]}\n"
                "- Still include relevant bullets in 'answer' for context, but use tables for structured lists\n"
                "- Tables are OPTIONAL - only use when data is clearly tabular\n\n"
                "CRITICAL RULES for answer bullets:\n"
                "1. Write in plain, professional language - explain what the contract says in simple terms\n"
                "2. Each bullet should be 1-2 sentences and directly answer the question\n"
                "3. CRITICAL: Each bullet MUST be at least 10 words long - provide complete, detailed information\n"
                "   - DO NOT include section headings, titles, or incomplete phrases like 'Computation of vacations.' or 'Additional vacation.'\n"
                "   - Each bullet must be a complete, meaningful sentence or sentences that fully explain the concept\n"
                "   - If you find headings or short phrases in the sources, expand them into full explanations\n"
                "4. ALWAYS start each bullet with a CAPITALIZED first word\n"
                "5. DO NOT copy raw text from clauses - SUMMARIZE and EXPLAIN the meaning in your own words\n"
                "   - Transform legal language into clear, actionable information\n"
                "   - Focus on the practical implications for workers\n"
                "   - Use simple, direct language that anyone can understand\n"
                "6. CRITICAL: ONLY USE SOURCES THAT DIRECTLY ANSWER THE QUESTION - Ignore irrelevant sources\n"
                "   - HOLIDAYS and VACATIONS are DIFFERENT concepts - do NOT confuse them\n"
                "   - If asked about HOLIDAYS (paid holidays like New Year's Day, Christmas), ONLY use holiday-related sources\n"
                "   - If asked about VACATIONS (time off with pay based on qualifying hours), ONLY use vacation-related sources\n"
                "   - If asked about guarantees, ONLY use guarantee-related sources\n"
                "   - DO NOT include unrelated sections even if they appear in sources\n"
                "   - If the question specifically asks for a list of holidays, DO NOT include vacation information\n"
                "7. DO NOT include:\n"
                "   - Timestamps, PDF metadata, long clause ID lists\n"
                "   - Incomplete sentences, raw contract text, or irrelevant content\n"
                "   - Hyphenation artifacts (e.g., fix 'combina - tion' to 'combination')\n"
                "   - Source citations, filenames, section numbers, or any parenthetical references\n"
                "8. CRITICAL: DO NOT include ANY source citations or references in your answer bullets\n"
                "   - NO filenames, NO section numbers, NO parenthetical citations\n"
                "   - Sources will be displayed separately in the UI\n"
                "   - Just focus on answering the question clearly without any citations\n"
                "9. Focus on WHAT the contract says and WHAT it means for workers\n"
                "   - Explain the key points clearly\n"
                "   - Make it actionable and understandable\n"
                "10. If the question asks about something specific, ONLY use sources that directly relate to that topic\n"
                "    - Example: If asked about 'holidays', search for sections about 'paid holidays' or 'holiday schedule', NOT vacation sections\n"
                "    - Example: If asked about 'vacations', search for sections about 'vacation pay' or 'qualifying hours', NOT holiday sections\n"
                "11. Group related information together - if multiple sources say similar things, combine them into one clear bullet\n"
                "12. Prioritize the most important information - lead with the key points\n"
                "13. If a bullet is too short (less than 10 words), expand it with more context and details from the contract\n\n"
                "Example of GOOD answer bullet:\n"
                '"Clerks are entitled to a 2-hour meal period, and if not sent to eat before the second hour begins, they must be paid for the work performed."\n\n'
                "Example of BAD answer bullet (DO NOT DO THIS):\n"
                '", the quarter-hour, the half-hour, the three-quarter hour or the even hour and time lost between the designated starting time and time turned to shall be deducted from the guarantee. (Pacific-Coast-Clerks-Contract-Document-2022-2028.pdf (2.4 (2.4,2.41,2.42,2.43,2.44,2.45,2.411,2.431,2.432,2.441,2.442,2.443,2.444,2.445,2.446,2.447,2.448,2.449,2.451,2.452,2.453,2.4441,2.4471,2.4491,2.4492) – HOURS AND SHIFTS))"\n\n'
                "Only use information from the provided clauses. Do NOT invent information.\n\n"
                f"User Question: {query}\n\n"
                f"Relevant Clauses:\n{context_snippets}\n\n"
                "Now provide your JSON response:"
            )
            try:
                raw_response = llm_client.generate(prompt, stream=True)
                json_payload = self._parse_llama_json(raw_response)
                if json_payload:
                    data = json_payload
                    raw_points = [
                        point.strip() for point in data.get("answer", []) if isinstance(point, str)
                    ]
                    # Clean up answer points: remove timestamps, long clause ID lists, formatting artifacts
                    answer_points = []
                    for point in raw_points:
                        cleaned = self._clean_answer_point(point)
                        # Ensure each bullet is at least 10 words long
                        if cleaned:
                            word_count = len(cleaned.split())
                            if word_count >= 10:
                                answer_points.append(cleaned)
                            else:
                                self.logger.debug(f"Filtered out short bullet ({word_count} words): {cleaned[:100]}")
                    # Extract tables if present
                    tables_data = data.get("tables", [])
                    if isinstance(tables_data, list):
                        # Validate table structure
                        validated_tables = []
                        for table in tables_data:
                            if isinstance(table, dict) and "headers" in table and "rows" in table:
                                if isinstance(table["headers"], list) and isinstance(table["rows"], list):
                                    validated_tables.append(table)
                        tables = validated_tables
                    disc = data.get("disclaimer")
                    if isinstance(disc, str) and disc.strip():
                        disclaimer = self._normalize_disclaimer(disc.strip())
            except Exception as exc:  # pragma: no cover
                print(f"[ContractInsightsEngine] Llama synthesis failed: {exc}")
                if len(sources) > 2:
                    try:
                        # Pre-clean excerpts for retry
                        cleaned_retry_sources = []
                        for src in sources[:2]:
                            excerpt = src.get('excerpt', '')[:800]
                            cleaned_excerpt = self._clean_excerpt_for_llm(excerpt)
                            if cleaned_excerpt:
                                cleaned_retry_sources.append(f"Source: {self._format_source_label(src)}\nClause: {cleaned_excerpt}")
                        retry_context = "\n\n".join(cleaned_retry_sources)
                        
                        retry_prompt = (
                            "You are a contract analyst for ILWU/PMA maritime agreements. "
                            "Your task is to provide clear, concise summaries that directly answer the user's question.\n\n"
                            "IMPORTANT: You MUST return valid JSON only. Do not include any text before or after the JSON.\n\n"
                            "JSON schema:\n"
                            "{\n"
                            '  "answer": ["First clear bullet point", "Second clear bullet point", "..."],\n'
                            '  "tables": [{"headers": ["Column1", "Column2"], "rows": [["Value1", "Value2"], ["Value3", "Value4"]]}],  // OPTIONAL: Use for structured data like holiday lists, benefit schedules, etc.\n'
                            '  "disclaimer": "Please refer to exact contract text for detailed understanding."\n'
                            "}\n\n"
                            "TABULATION RULES:\n"
                            "- If the question asks for lists, schedules, or tabular data (e.g., 'holiday list', 'vacation schedule', 'benefits'), use the 'tables' field\n"
                            "- Each table should have 'headers' (array of column names) and 'rows' (array of arrays, where each inner array is a row)\n"
                            "- Example for holidays: {\"headers\": [\"Holiday\", \"Date/Details\"], \"rows\": [[\"New Year's Day\", \"January 1\"], [\"Martin Luther King's Day\", \"Third Monday in January\"]]}\n"
                            "- Still include relevant bullets in 'answer' for context, but use tables for structured lists\n"
                            "- Tables are OPTIONAL - only use when data is clearly tabular\n\n"
                            "CRITICAL RULES for answer bullets:\n"
                            "1. Write in plain, professional language - explain what the contract says in simple terms\n"
                            "2. Each bullet should be 1-2 sentences and directly answer the question\n"
                            "3. CRITICAL: Each bullet MUST be at least 10 words long - provide complete, detailed information\n"
                            "   - DO NOT include section headings, titles, or incomplete phrases like 'Computation of vacations.' or 'Additional vacation.'\n"
                            "   - Each bullet must be a complete, meaningful sentence or sentences that fully explain the concept\n"
                            "   - If you find headings or short phrases in the sources, expand them into full explanations\n"
                            "4. ALWAYS start each bullet with a CAPITALIZED first word\n"
                            "5. DO NOT copy raw text from clauses - SUMMARIZE and EXPLAIN the meaning in your own words\n"
                            "   - Transform legal language into clear, actionable information\n"
                            "   - Focus on the practical implications for workers\n"
                            "   - Use simple, direct language that anyone can understand\n"
                            "6. CRITICAL: ONLY USE SOURCES THAT DIRECTLY ANSWER THE QUESTION - Ignore irrelevant sources\n"
                            "   - HOLIDAYS and VACATIONS are DIFFERENT concepts - do NOT confuse them\n"
                            "   - If asked about HOLIDAYS (paid holidays like New Year's Day, Christmas), ONLY use holiday-related sources\n"
                            "   - If asked about VACATIONS (time off with pay based on qualifying hours), ONLY use vacation-related sources\n"
                            "   - If asked about guarantees, ONLY use guarantee-related sources\n"
                            "   - DO NOT include unrelated sections even if they appear in sources\n"
                            "   - If the question specifically asks for a list of holidays, DO NOT include vacation information\n"
                            "7. DO NOT include:\n"
                            "   - Timestamps, PDF metadata, long clause ID lists\n"
                            "   - Incomplete sentences, raw contract text, or irrelevant content\n"
                            "   - Hyphenation artifacts (e.g., fix 'combina - tion' to 'combination')\n"
                            "   - Source citations, filenames, section numbers, or any parenthetical references\n"
                            "8. CRITICAL: DO NOT include ANY source citations or references in your answer bullets\n"
                            "   - NO filenames, NO section numbers, NO parenthetical citations\n"
                            "   - Sources will be displayed separately in the UI\n"
                            "   - Just focus on answering the question clearly without any citations\n"
                            "9. Focus on WHAT the contract says and WHAT it means for workers\n"
                            "   - Explain the key points clearly\n"
                            "   - Make it actionable and understandable\n"
                            "10. If the question asks about something specific, ONLY use sources that directly relate to that topic\n"
                            "    - Example: If asked about 'holidays', search for sections about 'paid holidays' or 'holiday schedule', NOT vacation sections\n"
                            "    - Example: If asked about 'vacations', search for sections about 'vacation pay' or 'qualifying hours', NOT holiday sections\n"
                            "11. Group related information together - if multiple sources say similar things, combine them into one clear bullet\n"
                            "12. Prioritize the most important information - lead with the key points\n"
                            "13. If a bullet is too short (less than 10 words), expand it with more context and details from the contract\n\n"
                            "Example of GOOD answer bullet:\n"
                            '"Clerks are entitled to a 2-hour meal period, and if not sent to eat before the second hour begins, they must be paid for the work performed."\n\n'
                            "Example of BAD answer bullet (DO NOT DO THIS):\n"
                            '", the quarter-hour, the half-hour, the three-quarter hour or the even hour and time lost between the designated starting time and time turned to shall be deducted from the guarantee. (Pacific-Coast-Clerks-Contract-Document-2022-2028.pdf (2.4 (2.4,2.41,2.42,2.43,2.44,2.45,2.411,2.431,2.432,2.441,2.442,2.443,2.444,2.445,2.446,2.447,2.448,2.449,2.451,2.452,2.453,2.4441,2.4471,2.4491,2.4492) – HOURS AND SHIFTS))"\n\n'
                            "Only use information from the provided clauses. Do NOT invent information.\n\n"
                            f"User Question: {query}\n\n"
                            f"Relevant Clauses:\n{retry_context}\n\n"
                            "Now provide your JSON response:"
                        )
                        retry_response = llm_client.generate(
                            retry_prompt,
                            timeout=max(20, Config.LLAMA_TIMEOUT // 2),
                            stream=True,
                        )
                        retry_data = self._parse_llama_json(retry_response)
                        if retry_data:
                            raw_points = [
                                point.strip()
                                for point in retry_data.get("answer", [])
                                if isinstance(point, str)
                            ]
                            # Clean up answer points
                            answer_points = []
                            for point in raw_points:
                                cleaned = self._clean_answer_point(point)
                                # Ensure each bullet is at least 10 words long
                                if cleaned:
                                    word_count = len(cleaned.split())
                                    if word_count >= 10:
                                        answer_points.append(cleaned)
                                    else:
                                        self.logger.debug(f"Filtered out short bullet ({word_count} words): {cleaned[:100]}")
                            # Extract tables if present from retry
                            retry_tables_data = retry_data.get("tables", [])
                            if isinstance(retry_tables_data, list):
                                # Validate table structure
                                validated_retry_tables = []
                                for table in retry_tables_data:
                                    if isinstance(table, dict) and "headers" in table and "rows" in table:
                                        if isinstance(table["headers"], list) and isinstance(table["rows"], list):
                                            validated_retry_tables.append(table)
                                if validated_retry_tables:
                                    tables = validated_retry_tables
                            disc = retry_data.get("disclaimer")
                            if isinstance(disc, str) and disc.strip():
                                disclaimer = self._normalize_disclaimer(disc.strip())
                    except Exception as retry_exc:  # pragma: no cover
                        print(f"[ContractInsightsEngine] Reduced-context Llama retry failed: {retry_exc}")

        if not answer_points:
            heuristic_points = self._build_fallback_summary(query, sources)
            # Validate and filter fallback bullets (must be at least 10 words)
            for point in heuristic_points:
                cleaned = self._clean_answer_point(point)
                if cleaned:
                    word_count = len(cleaned.split())
                    if word_count >= 10:
                        answer_points.append(cleaned)
                    else:
                        self.logger.debug(f"Filtered out short fallback bullet ({word_count} words): {cleaned[:100]}")

        if not answer_points:
            # Extract excerpts but ensure they're at least 10 words
            for src in sources[:4]:
                excerpt = src.get("excerpt", "")
                if excerpt:
                    cleaned_excerpt = self._clean_excerpt_for_llm(excerpt)
                    cleaned = self._clean_answer_point(cleaned_excerpt[:240])
                    if cleaned:
                        word_count = len(cleaned.split())
                        if word_count >= 10:
                            answer_points.append(cleaned)
        if not answer_points:
            answer_points = ["No synthesized answer available from the retrieved clauses."]

        return answer_points, disclaimer, tables

    def _assemble_contract_content(
        self, answer_points: List[str], disclaimer: str, sources: List[Dict[str, Any]]
    ) -> str:
        # Just return the answer points without sources - sources are displayed separately in the UI
        answer_section = "\n".join(f"- {point}" for point in answer_points)
        return (
            f"{answer_section}\n\n"
            f"Disclaimer: {disclaimer}"
        )

    @staticmethod
    def _format_source_label(source_entry: Dict[str, Any]) -> str:
        source = source_entry.get("source", "unknown")
        page = source_entry.get("page")
        section_heading = source_entry.get("section_heading") or source_entry.get("section_title")
        clause = source_entry.get("clause")
        main_clause_id = source_entry.get("main_clause_id")
        clause_heading = source_entry.get("clause_heading")

        # Use main_clause_id if available, otherwise fall back to clause
        # Do NOT include sub_clause_ids - just show the main clause ID
        clause_label = main_clause_id or (clause if clause != "intro" else None)

        if clause_label:
            label_parts = [clause_label]
            if clause_heading:
                label_parts.append(clause_heading)
            elif section_heading:
                label_parts.append(section_heading)
            return f"{source} ({' – '.join(label_parts)})"

        if section_heading:
            return f"{source} ({section_heading})"

        if page is not None:
            return f"{source} (page {page})"
        return source

    def _generate_wage_sql(self, question: str, mode: str = "llama") -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate SQL and return metadata about the query."""
        llm_client = self._get_llm_client(mode)
        api_name = "OpenAI" if mode == "openai" else "Llama"
        self.logger.info(f"[Wage SQL Generation] Using {api_name} API to generate SQL query")
        if not llm_client.available():
            return None, {}

        schema_desc = self.wage_schema or ""
        value_context: List[str] = []
        employee_types = self.wage_metadata.get("employee_types")
        if employee_types:
            value_context.append(
                "Valid EmployeeType values: " + ", ".join(employee_types)
            )
            value_context.append("If the request references foremen or walking bosses, use EmployeeType 'walking_bosses'.")
        skill_types = self.wage_metadata.get("skill_types")
        if skill_types:
            sample = ", ".join(skill_types[:8]) + (" ..." if len(skill_types) > 8 else "")
            value_context.append("Example SkillType values: " + sample)

        prepared_question, metadata = self._prepare_wage_question(question)
        
        # Add default filter rules - but prioritize user-specified values
        default_rules = (
            "\nCRITICAL FILTER RULES:\n"
            "- ALWAYS include WHERE clause with EmployeeType, SkillType, and ExperienceHours filters\n"
            "- CRITICAL: If the user explicitly mentions SkillType or ExperienceHours, use that value directly. DO NOT override with defaults\n"
            "\nVALID SkillType VALUES:\n"
            "- For longshoremen/clerks: 'Basic', 'Skill I', 'Skill II', 'Skill III'\n"
            "- For clerks only: 'Kitchen/Tower/Computer Clerk', 'Clerk Supervisor', 'Chief Supervisor & Supercargo'\n"
            "- For walking_bosses: 'All'\n"
            "\nSKILL TYPE MAPPING:\n"
            "- User says 'Skill 1' or 'Skill I' → Use SkillType = 'Skill I'\n"
            "- User says 'Skill 2' or 'Skill II' → Use SkillType = 'Skill II'\n"
            "- User says 'Skill 3' or 'Skill III' → Use SkillType = 'Skill III'\n"
            "- User says 'Basic' → Use SkillType = 'Basic'\n"
            "\nVALID ExperienceHours VALUES:\n"
            "- For clerks/longshoremen: 'All', '4001+', '2001-4000', '1001-2000', '0-1000'\n"
            "- For walking_bosses: 'All'\n"
            "\nEXPERIENCE HOURS MAPPING:\n"
            "- User says '<1000 hours' or 'less than 1000 hours' → Use ExperienceHours = '0-1000'\n"
            "- User says '0-1000 hours' → Use ExperienceHours = '0-1000'\n"
            "- User says '1001-2000 hours' → Use ExperienceHours = '1001-2000'\n"
            "- User says '2001-4000 hours' → Use ExperienceHours = '2001-4000'\n"
            "- User says '4001+' or 'more than 4000 hours' → Use ExperienceHours = '4001+'\n"
            "- User says 'all' for experience → Use ExperienceHours = 'All'\n"
            "\nDEFAULT VALUES (only use when NOT explicitly specified by user):\n"
            "  - For 'walking_bosses': If SkillType is not specified, use SkillType = 'All'\n"
            "  - For 'walking_bosses': If ExperienceHours is not specified, use ExperienceHours = 'All'\n"
            "  - For 'longshoremen' or 'clerks': If SkillType is not specified, use SkillType = 'Basic'\n"
            "  - For 'longshoremen' or 'clerks': If ExperienceHours is not specified, use ExperienceHours = '0-1000'\n"
        )
        
        prompt = (
            "You are an assistant that writes MySQL SELECT queries for the wage schedule table.\n"
            f"Table name: {WAGE_TABLE_NAME}\n"
            f"Columns:\n{schema_desc}\n\n"
            "Rules:\n"
            "- Generate a single SELECT statement.\n"
            "- SELECT only the columns needed: EmployeeType, SkillType, ExperienceHours, FiscalYear, and the relevant shift columns (Shift1, Shift2, Shift3, Shift1_2_Overtime, Shift3_Overtime)\n"
            "- DO NOT use SELECT * - explicitly list the columns you need\n"
            "- Infer filters (employee type, skill level, fiscal year, shift) from the question.\n"
            "- Limit the result to 100 rows unless aggregation is required.\n"
            "- Use existing columns directly; only perform arithmetic if the user explicitly asks for it.\n"
            "- When the user mentions first/second/third shift, map them to columns Shift1, Shift2, Shift3 respectively.\n"
            "- Do not include comments or explanations.\n"
            "- Return only the SQL statement starting with SELECT; do not prefix it with words like 'SQL:' or 'SELECT query:'.\n"
            "- CRITICAL: Use actual literal values in WHERE clauses (e.g., SkillType = 'All'), NOT parameter placeholders like '?' or ':param'\n"
            "- All values must be properly quoted strings (e.g., 'walking_bosses', 'All', 'Basic') or unquoted numbers (e.g., 2022, 2027)\n"
            "- CRITICAL: DO NOT include any conditions on EndDate or StartDate in WHERE clause\n"
            "- CRITICAL: FiscalYear is already a numeric year value (e.g., 2025, 2026), NOT a date. Use direct numeric comparisons like FiscalYear >= 2025 or FiscalYear = 2025. DO NOT use EXTRACT(YEAR FROM FiscalYear) or any date functions on FiscalYear\n"
            "- CRITICAL: Only use FiscalYear for date filtering (e.g., FiscalYear >= 2022 AND FiscalYear <= 2027 or FiscalYear = 2025)\n"
            "- CRITICAL: The CURRENT YEAR is 2025. When the user asks for 'current year', 'latest', 'most recent', or similar terms, use FiscalYear = 2025\n"
            + default_rules
        )
        if value_context:
            prompt += "\nAdditional context:\n" + "\n".join(f"- {line}" for line in value_context) + "\n"
        prompt += f"\nQuestion: {prepared_question}\n\nSQL:"
        
        try:
            self.logger.debug(f"[Wage SQL Generation] Prompt length: {len(prompt)} characters")
            raw_sql = llm_client.generate(prompt, stream=True)
            self.logger.debug(f"[Wage SQL Generation] Raw LLM response length: {len(raw_sql)} characters")
            self.logger.debug(f"[Wage SQL Generation] Raw LLM response (first 500 chars): {raw_sql[:500]}")
            
            if not raw_sql or not raw_sql.strip():
                self.logger.warning("[Wage SQL Generation] LLM returned empty response")
                return None, metadata
            
            sql = self._extract_sql_statement(raw_sql)
            if not sql:
                self.logger.warning(
                    "[Wage SQL Generation] Failed to extract SQL from LLM response. "
                    f"Response (first 500 chars): {raw_sql[:500]}"
                )
                return None, metadata
            
            self.logger.info(f"[Wage SQL Generation] Successfully extracted SQL: {sql[:200]}...")
            return sql, metadata
        except Exception as exc:
            self.logger.error(
                f"[Wage SQL Generation] Exception during SQL generation: {exc}",
                exc_info=True
            )
            return None, metadata

    def _prepare_wage_question(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """Prepare question with default filters and return metadata about shift mentions."""
        lower = question.lower()
        notes: List[str] = []
        metadata: Dict[str, Any] = {
            "shift_mentioned": False,
            "shift_columns": [],
            "employee_type": None,
        }

        # Detect shift mentions
        if "first shift" in lower or "shift 1" in lower:
            notes.append("Focus on the Shift1 column (first shift wages).")
            metadata["shift_mentioned"] = True
            metadata["shift_columns"] = ["Shift1"]
        if "second shift" in lower or "shift 2" in lower:
            notes.append("Include the Shift2 column (second shift wages).")
            metadata["shift_mentioned"] = True
            if "Shift2" not in metadata["shift_columns"]:
                metadata["shift_columns"].append("Shift2")
        if "third shift" in lower or "shift 3" in lower:
            notes.append("Include the Shift3 column (third shift wages).")
            metadata["shift_mentioned"] = True
            if "Shift3" not in metadata["shift_columns"]:
                metadata["shift_columns"].append("Shift3")

        # Detect current year mentions
        current_year_terms = ["current year", "latest", "most recent", "this year", "present year", "now"]
        if any(term in lower for term in current_year_terms):
            notes.append("The user is asking for current year data. Use FiscalYear = 2025 (the current year is 2025).")
            metadata["current_year_mentioned"] = True
        else:
            metadata["current_year_mentioned"] = False
        
        # Detect if SkillType is explicitly mentioned
        # Patterns for longshoremen/clerks: Basic, Skill I, Skill II, Skill III
        # Patterns for clerks: Kitchen/Tower/Computer Clerk, Chief Supervisor & Supercargo, Clerk Supervisor
        # Pattern for walking bosses: All
        skill_type_patterns = [
            r"skill\s*[123]", r"skill\s*[i]{1,3}", r"skill\s+[i]{1,3}",  # Skill 1, Skill 2, Skill I, Skill II, Skill III
            r"skill\s*type", "basic", "advanced", "intermediate", 
            "skilled", "unskilled", "all skill", "skill level", "skill tier",
            "kitchen", "tower", "computer clerk", "chief supervisor", "supercargo", "clerk supervisor",
            "supervisor"
        ]
        skill_type_mentioned = any(re.search(pattern, lower) for pattern in skill_type_patterns)
        metadata["skill_type_mentioned"] = skill_type_mentioned
        
        # Detect specific skill type value if mentioned
        detected_skill_type = None
        if re.search(r"skill\s*2|skill\s*ii", lower, re.IGNORECASE):
            detected_skill_type = "Skill II"
            notes.append("User mentioned 'Skill 2' or 'Skill II'. Use SkillType = 'Skill II' for longshoremen/clerks.")
        elif re.search(r"skill\s*1|skill\s*i\b", lower, re.IGNORECASE) and not re.search(r"skill\s*[23]|skill\s*[i]{2,3}", lower, re.IGNORECASE):
            detected_skill_type = "Skill I"
            notes.append("User mentioned 'Skill 1' or 'Skill I'. Use SkillType = 'Skill I' for longshoremen/clerks.")
        elif re.search(r"skill\s*3|skill\s*iii", lower, re.IGNORECASE):
            detected_skill_type = "Skill III"
            notes.append("User mentioned 'Skill 3' or 'Skill III'. Use SkillType = 'Skill III' for longshoremen/clerks.")
        elif "basic" in lower:
            detected_skill_type = "Basic"
            notes.append("User mentioned 'Basic'. Use SkillType = 'Basic' for longshoremen/clerks.")
        elif any(term in lower for term in ["kitchen", "tower", "computer clerk"]):
            detected_skill_type = "Kitchen/Tower/Computer Clerk"
            notes.append("User mentioned kitchen/tower/computer clerk. Use SkillType = 'Kitchen/Tower/Computer Clerk' for clerks.")
        elif "chief supervisor" in lower or "supercargo" in lower:
            detected_skill_type = "Chief Supervisor & Supercargo"
            notes.append("User mentioned chief supervisor or supercargo. Use SkillType = 'Chief Supervisor & Supercargo' for clerks.")
        elif "clerk supervisor" in lower:
            detected_skill_type = "Clerk Supervisor"
            notes.append("User mentioned clerk supervisor. Use SkillType = 'Clerk Supervisor' for clerks.")
        
        if detected_skill_type:
            metadata["detected_skill_type"] = detected_skill_type
        
        # Detect if ExperienceHours is explicitly mentioned
        # Valid ExperienceHours values: "All", "4001+", "2001-4000", "1001-2000", "0-1000"
        # Look for patterns that clearly indicate experience hours (not fiscal years)
        experience_patterns = [
            r"<[0-9]+.*hour", r"less\s+than\s+[0-9]+.*hour", r">[0-9]+.*hour", r"more\s+than\s+[0-9]+.*hour",
            r"[0-9]+\s*-\s*[0-9]+.*hour", r"[0-9]+.*hour.*experience", r"experience.*hour",
            r"[0-9]+.*hours.*experience", r"hours\s+of\s+experience", r"experience\s+hours",
            r"[0-9]+.*hours.*of", r"hour.*experience"
        ]
        experience_mentioned = any(re.search(pattern, lower) for pattern in experience_patterns)
        # Exclude fiscal year references (FY25, FY2025, fiscal year 2025, etc.)
        if experience_mentioned:
            # Check if it's actually a fiscal year reference
            fiscal_year_patterns = [r"\bfy\s*[0-9]+", r"fiscal\s+year"]
            if any(re.search(pattern, lower) for pattern in fiscal_year_patterns):
                # If fiscal year is mentioned, check if the hour pattern is near it
                # If user says "FY25 to FY27" that's not experience hours
                if "experience" not in lower:
                    experience_mentioned = False
        metadata["experience_mentioned"] = experience_mentioned
        
        # Detect and map specific ExperienceHours value if mentioned
        detected_experience_hours = None
        # Check for <1000 or less than 1000 hours
        if re.search(r"<[^0-9]*1000|<1000|less\s+than\s+1000", lower, re.IGNORECASE):
            detected_experience_hours = "0-1000"
            notes.append("User mentioned '<1000 hours' or 'less than 1000 hours'. Use ExperienceHours = '0-1000' for clerks/longshoremen.")
        # Check for 0-1000 range
        elif re.search(r"\b0\s*[-~]\s*1000|\b1000\s*[-~]\s*0", lower, re.IGNORECASE):
            detected_experience_hours = "0-1000"
            notes.append("User mentioned '0-1000 hours'. Use ExperienceHours = '0-1000' for clerks/longshoremen.")
        # Check for 1001-2000 range
        elif re.search(r"\b1001\s*[-~]\s*2000|\b2000\s*[-~]\s*1001", lower, re.IGNORECASE) or re.search(r"between\s+1001\s+and\s+2000", lower, re.IGNORECASE):
            detected_experience_hours = "1001-2000"
            notes.append("User mentioned '1001-2000 hours'. Use ExperienceHours = '1001-2000' for clerks/longshoremen.")
        # Check for 2001-4000 range
        elif re.search(r"\b2001\s*[-~]\s*4000|\b4000\s*[-~]\s*2001", lower, re.IGNORECASE) or re.search(r"between\s+2001\s+and\s+4000", lower, re.IGNORECASE):
            detected_experience_hours = "2001-4000"
            notes.append("User mentioned '2001-4000 hours'. Use ExperienceHours = '2001-4000' for clerks/longshoremen.")
        # Check for 4001+ or more than 4000
        elif re.search(r"4001\+|>\s*4000|more\s+than\s+4000|4000\+", lower, re.IGNORECASE):
            detected_experience_hours = "4001+"
            notes.append("User mentioned '4001+' or 'more than 4000 hours'. Use ExperienceHours = '4001+' for clerks/longshoremen.")
        # Check for "all" experience
        elif "all" in lower and ("experience" in lower or "hour" in lower):
            detected_experience_hours = "All"
            notes.append("User mentioned 'all' for experience hours. Use ExperienceHours = 'All'.")
        
        if detected_experience_hours:
            metadata["detected_experience_hours"] = detected_experience_hours
        
        # Detect employee type
        if "walking boss" in lower or "walking bosses" in lower or "foremen" in lower or "foreman" in lower:
            notes.append("Map mentions of foremen/walking bosses to EmployeeType 'walking_bosses'.")
            metadata["employee_type"] = "walking_bosses"
            if not skill_type_mentioned:
                notes.append("For walking_bosses: If SkillType is not specified, default to 'All'.")
            if not experience_mentioned:
                notes.append("For walking_bosses: If ExperienceHours is not specified, default to 'All'.")
        elif "longshoremen" in lower or "longshore" in lower:
            metadata["employee_type"] = "longshoremen"
            if not skill_type_mentioned:
                notes.append("If SkillType is not specified, default to 'Basic'.")
            if not experience_mentioned:
                notes.append("If ExperienceHours is not specified, default to '0-1000'.")
        elif "clerk" in lower or "clerks" in lower:
            metadata["employee_type"] = "clerks"
            if not skill_type_mentioned:
                notes.append("If SkillType is not specified, default to 'Basic'.")
            if not experience_mentioned:
                notes.append("If ExperienceHours is not specified, default to '0-1000'.")

        # Add notes for explicitly mentioned values (prioritize these)
        if skill_type_mentioned:
            notes.insert(0, "IMPORTANT: The user has explicitly specified a SkillType value in their query. You MUST use that value directly and NOT override it with any default value.")
        if experience_mentioned:
            notes.insert(0, "IMPORTANT: The user has explicitly specified ExperienceHours in their query. You MUST use that value directly and NOT override it with any default value.")
        
        # Default filters - only apply if not explicitly mentioned
        if metadata["employee_type"]:
            notes.append("ALWAYS include WHERE clause with filters for EmployeeType, SkillType, and ExperienceHours.")
            if metadata["employee_type"] == "walking_bosses":
                if not skill_type_mentioned:
                    notes.append("For walking_bosses: If user doesn't specify SkillType, use 'All'.")
                if not experience_mentioned:
                    notes.append("For walking_bosses: If user doesn't specify ExperienceHours, use 'All'.")
            elif metadata["employee_type"] in ["longshoremen", "clerks"]:
                if not skill_type_mentioned:
                    notes.append("If user doesn't specify SkillType, use 'Basic'.")
                if not experience_mentioned:
                    notes.append("If user doesn't specify ExperienceHours, use '0-1000'.")

        if notes:
            return f"{question}\n\nAdditional guidance for SQL generator:\n" + "\n".join(f"- {note}" for note in notes), metadata
        return question, metadata

    @staticmethod
    def _extract_sql_statement(text: str) -> Optional[str]:
        if not text:
            return None
        cleaned = text.strip()
        cleaned = re.sub(r"```sql", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "").strip()
        match = re.search(r"SELECT\s.+", cleaned, re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        sql = match.group(0)
        # Strip leading descriptors like "SELECT query:" or "SQL:"
        sql = re.sub(r"^(SELECT\s+query\s*:\s*)", "", sql, flags=re.IGNORECASE).lstrip()
        sql = re.sub(r"^(SQL\s*:\s*)", "", sql, flags=re.IGNORECASE).lstrip()
        # If residual descriptor remains before the actual SELECT, grab the next occurrence.
        if not sql.lower().startswith("select"):
            second_match = re.search(r"SELECT\s.+", sql, re.IGNORECASE | re.DOTALL)
            if second_match:
                sql = second_match.group(0).lstrip()
            else:
                return None
        sql = sql.split(";")[0]
        sql = sql.strip()
        
        # Remove parameter placeholders (?, :param, etc.) and replace with appropriate defaults
        # Replace ? with NULL or remove the condition if it's in an OR clause
        # Pattern: (SkillType = 'All' OR SkillType = ?) -> (SkillType = 'All')
        sql = re.sub(r"\(\s*(\w+)\s*=\s*'([^']+)'\s+OR\s+\1\s*=\s*\?\s*\)", r"(\1 = '\2')", sql, flags=re.IGNORECASE)
        sql = re.sub(r"(\w+)\s*=\s*\?", r"\1 IS NOT NULL", sql, flags=re.IGNORECASE)
        sql = re.sub(r"(\w+)\s*=\s*:\w+", r"\1 IS NOT NULL", sql, flags=re.IGNORECASE)
        
        # Remove EndDate and StartDate conditions from WHERE clause
        # Remove conditions like "AND EndDate IS NULL" or "AND EndDate = ..."
        sql = re.sub(r"\s+AND\s+EndDate\s+[^\s]+(?:\s+[^\s]+)*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\s+AND\s+StartDate\s+[^\s]+(?:\s+[^\s]+)*", "", sql, flags=re.IGNORECASE)
        # Handle cases where EndDate/StartDate is the first condition after WHERE
        sql = re.sub(r"WHERE\s+EndDate\s+[^\s]+(?:\s+[^\s]+)*\s+AND", "WHERE", sql, flags=re.IGNORECASE)
        sql = re.sub(r"WHERE\s+StartDate\s+[^\s]+(?:\s+[^\s]+)*\s+AND", "WHERE", sql, flags=re.IGNORECASE)
        # Handle cases where EndDate/StartDate is the only condition
        sql = re.sub(r"WHERE\s+(EndDate|StartDate)\s+[^\s]+(?:\s+[^\s]+)*$", "", sql, flags=re.IGNORECASE)
        # Clean up any double spaces or trailing AND/WHERE
        sql = re.sub(r"\s+", " ", sql)
        sql = re.sub(r"\s+AND\s+$", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"WHERE\s+$", "", sql, flags=re.IGNORECASE)
        
        return sql

    def _answer_wage_schedule(self, query: str, debug_info: Dict[str, Any], mode: str = "llama") -> Dict[str, Any]:
        if not self.wage_db_conn or not self.wage_schema:
            message = "Wage schedule data is not currently available."
            debug_info["wage_sql"] = None
            return {
                "response_type": "wage_schedule_sql",
                "content": message,
                "results": [],
                "sql_query": None,
                "matches": [],
                "sources": [],
                "total_matches": 0,
                "answer_points": [],
                "disclaimer": None,
                "opening": f"Wage schedule details for '{query}'",
            }

        sql_query, query_metadata = self._generate_wage_sql(query, mode=mode)
        if not sql_query:
            message = "I couldn't generate a wage schedule query from that question."
            debug_info["wage_sql"] = None
            debug_info["wage_sql_error"] = "SQL generation returned None - check logs for details"
            self.logger.warning(
                "Wage schedule SQL generation failed for query: %s (mode: %s, metadata: %s)",
                query, mode, query_metadata
            )
            return {
                "response_type": "wage_schedule_sql",
                "content": message,
                "results": [],
                "sql_query": None,
                "matches": [],
                "sources": [],
                "total_matches": 0,
                "answer_points": [],
                "disclaimer": None,
                "opening": f"Wage schedule details for '{query}'",
            }

        # Log the generated SQL query
        self.logger.info("Wage schedule SQL generated for query: %s", query)
        self.logger.info("Generated SQL: %s", sql_query)
        self.logger.info("Query metadata: %s", query_metadata)

        try:
            self.logger.info("Executing wage schedule SQL query...")
            df = pd.read_sql(sql_query, self.wage_db_conn)
            self.logger.info("Wage schedule SQL executed successfully. Rows returned: %d", len(df))
        except mysql.connector.Error as db_exc:
            # Database-specific errors
            self.logger.error("Database error executing wage schedule SQL: %s", db_exc)
            debug_info["wage_sql"] = sql_query
            debug_info["sql_error"] = str(db_exc)
            error_msg = "We aren't able to process the wage schedule query. Please try rephrasing your question or check if the requested data is available."
            return {
                "response_type": "wage_schedule_sql",
                "content": error_msg,
                "results": [],
                "sql_query": sql_query,
                "matches": [],
                "sources": [],
                "total_matches": 0,
                "answer_points": [error_msg],
                "disclaimer": None,
                "opening": f"Wage schedule details for '{query}'",
            }
        except Exception as exc:
            # Generic errors
            self.logger.error("Error executing wage schedule SQL: %s", exc)
            debug_info["wage_sql"] = sql_query
            debug_info["sql_error"] = str(exc)
            error_msg = "We aren't able to process the wage schedule query. Please try rephrasing your question."
            return {
                "response_type": "wage_schedule_sql",
                "content": error_msg,
                "results": [],
                "sql_query": sql_query,
                "matches": [],
                "sources": [],
                "total_matches": 0,
                "answer_points": [error_msg],
                "disclaimer": None,
                "opening": f"Wage schedule details for '{query}'",
            }

        debug_info["wage_sql"] = sql_query
        debug_info["row_count"] = len(df)
        
        # Log final results summary
        if not df.empty:
            self.logger.info("Wage schedule results summary: %d rows, columns: %s", len(df), list(df.columns))
            if "FiscalYear" in df.columns:
                years = sorted(df["FiscalYear"].unique())
                self.logger.info("Fiscal years in results: %s", years)
            if "EmployeeType" in df.columns:
                employee_types = df["EmployeeType"].unique()
                self.logger.info("Employee types in results: %s", list(employee_types))
        
        # Detect if this is an analytical question that needs LLM summarization
        is_analytical = self._is_analytical_wage_question(query)
        
        # Generate summary points - use LLM for analytical questions, min/max for simple queries
        analysis_tables: List[Dict[str, Any]] = []
        if df.empty:
            summary_points: List[str] = ["No wage records matched the filters."]
        elif is_analytical:
            # Use LLM to generate analytical insights
            summary_points, llm_tables = self._synthesize_wage_analysis(query, df, mode=mode)
            # If LLM fails, fall back to min/max
            if not summary_points:
                summary_points = self._build_wage_summary_min_max(df, query)
            else:
                # Use LLM-generated tables if available, otherwise generate default tables
                analysis_tables = llm_tables if llm_tables else self._generate_wage_analysis_tables(df, query)
        else:
            # Simple queries just get min/max summary
            summary_points = self._build_wage_summary_min_max(df, query)
        
        # Create opening message
        if not df.empty:
            opening = "Based on the data:"
        else:
            opening = f"Wage schedule details for '{query}'"
        
        # Build simple content with SQL query
        content_parts = [
            f"SQL Query:\n```sql\n{sql_query}\n```",
        ]
        
        if df.empty:
            content_parts.append("\nNo rows returned for the requested criteria.")
        
        # Convert to dict for frontend and rename all shift columns to append "Hourly Rate"
        df_display = df.copy()
        # Rename all shift-related columns to append "Hourly Rate"
        shift_column_mapping = {
            "Shift1": "Shift1 Hourly Rate",
            "Shift2": "Shift2 Hourly Rate",
            "Shift3": "Shift3 Hourly Rate",
            "Shift1_2_Overtime": "Shift1_2_Overtime Hourly Rate",
            "Shift3_Overtime": "Shift3_Overtime Hourly Rate",
        }
        for old_name, new_name in shift_column_mapping.items():
            if old_name in df_display.columns:
                df_display = df_display.rename(columns={old_name: new_name})
        # Convert FiscalYear to string for display
        if "FiscalYear" in df_display.columns:
            df_display["FiscalYear"] = df_display["FiscalYear"].astype(str)
        results_data = df_display.to_dict(orient="records")
        
        # Generate chart data (bar chart for single year, line chart for multiple years)
        chart_data = None
        chart_type = None
        if not df.empty:
            chart_result = self._prepare_wage_chart(df, query_metadata)
            if chart_result:
                chart_data = chart_result
                chart_type = chart_result.get("chart_type", "line")
                # Update chart series names to use renamed column names with "Hourly Rate"
                shift_column_mapping = {
                    "Shift1": "Shift1 Hourly Rate",
                    "Shift2": "Shift2 Hourly Rate",
                    "Shift3": "Shift3 Hourly Rate",
                    "Shift1_2_Overtime": "Shift1_2_Overtime Hourly Rate",
                    "Shift3_Overtime": "Shift3_Overtime Hourly Rate",
                }
                if "series" in chart_data:
                    if isinstance(chart_data["series"], list):
                        for series in chart_data["series"]:
                            if "name" in series:
                                original_name = series["name"]
                                if original_name in shift_column_mapping:
                                    series["name"] = shift_column_mapping[original_name]
        
        # Log the results dataset for debugging
        self.logger.info("Results dataset being sent to frontend:")
        self.logger.info("  - Number of rows: %d", len(results_data))
        if results_data:
            self.logger.info("  - First row keys: %s", list(results_data[0].keys()))
            self.logger.info("  - First row sample: %s", results_data[0])
            if len(results_data) > 1:
                self.logger.info("  - Second row sample: %s", results_data[1])
        
        # Pass raw data to frontend - let it handle table and chart rendering
        payload = {
            "response_type": "wage_schedule_sql",
            "content": "\n".join(content_parts),
            "results": results_data,  # Raw data for frontend to render
            "sql_query": sql_query,
            "matches": [],
            "sources": [],
            "total_matches": len(df),
            "answer_points": summary_points,
            "tables": analysis_tables if analysis_tables else [],  # Add analytical summary tables
            "disclaimer": "Please refer to exact contract text for detailed understanding.",
            "opening": opening,
        }
        
        # Add chart data if available
        if chart_data:
            payload["chart"] = chart_data
            payload["chart_type"] = chart_type
        
        # Log the full payload structure
        self.logger.info("Payload structure: response_type=%s, results_count=%d, answer_points_count=%d, opening=%s",
                        payload["response_type"], len(payload["results"]), len(payload["answer_points"]), payload["opening"])
            
        return payload

    def _build_wage_summary(self, df: pd.DataFrame, question: str) -> List[str]:
        summary: List[str] = []
        focus_shift = None
        question_lower = question.lower()
        if "shift1" in df.columns:
            if any(term in question_lower for term in ["shift1", "first shift"]):
                focus_shift = "Shift1"
            elif any(term in question_lower for term in ["shift2", "second shift"]):
                focus_shift = "Shift2" if "Shift2" in df.columns else None
            elif any(term in question_lower for term in ["shift3", "third shift"]):
                focus_shift = "Shift3" if "Shift3" in df.columns else None

        numeric_columns = [col for col in df.columns if df[col].dtype in ("float64", "int64")]
        target_columns = [focus_shift] if focus_shift else []
        target_columns += [col for col in ["Shift1", "Shift2", "Shift3"] if col in df.columns and col != focus_shift]
        target_columns = [col for col in target_columns if col in numeric_columns]

        fiscal_col = "FiscalYear" if "FiscalYear" in df.columns else None
        employee_col = "EmployeeType" if "EmployeeType" in df.columns else None
        skill_col = "SkillType" if "SkillType" in df.columns else None

        if target_columns and fiscal_col:
            focus_col = target_columns[0]
            yearly = (
                df.groupby(fiscal_col)[focus_col]
                .agg(["mean", "min", "max"])
                .round(2)
                .reset_index()
            )
            yearly_summary = ", ".join(
                f"{row[fiscal_col]}: avg {row['mean']:.2f}, range {row['min']:.2f}-{row['max']:.2f}"
                for _, row in yearly.head(5).iterrows()
            )
            summary.append(
                f"{focus_col} trend by fiscal year — {yearly_summary}"
            )

        if employee_col and target_columns:
            focus_col = target_columns[0]
            by_employee = (
                df.groupby(employee_col)[focus_col]
                .mean()
                .round(2)
                .sort_values(ascending=False)
                .head(5)
            )
            summary.append(
                f"Top {min(5, len(by_employee))} employee types by average {focus_col}: "
                + ", ".join(f"{idx}: {val:.2f}" for idx, val in by_employee.items())
            )

        if skill_col and target_columns:
            focus_col = target_columns[0]
            by_skill = (
                df.groupby(skill_col)[focus_col]
                .mean()
                .round(2)
                .sort_values(ascending=False)
                .head(5)
            )
            summary.append(
                f"Top {min(5, len(by_skill))} skill tiers by average {focus_col}: "
                + ", ".join(f"{idx}: {val:.2f}" for idx, val in by_skill.items())
            )

        if not summary and numeric_columns:
            focus_col = numeric_columns[0]
            summary.append(
                f"Average {focus_col}: {df[focus_col].mean():.2f} "
                f"(min {df[focus_col].min():.2f}, max {df[focus_col].max():.2f})"
            )

        return summary

    def _is_analytical_wage_question(self, query: str) -> bool:
        """Check if the query asks for analytical insights like 'which', 'highest', 'lowest', etc."""
        query_lower = query.lower()
        analytical_keywords = [
            "which", "who", "what", "highest", "lowest", "best", "worst",
            "top", "bottom", "most", "least", "compare", "comparison",
            "difference", "different", "rank", "ranking", "order by",
            "maximum", "minimum", "max", "min"
        ]
        return any(keyword in query_lower for keyword in analytical_keywords)
    
    def _synthesize_wage_analysis(
        self, 
        query: str, 
        df: pd.DataFrame, 
        mode: str = "llama"
    ) -> List[str]:
        """Use LLM to generate analytical insights from wage data. Returns (answer_points, tables)."""
        if df.empty:
            return ["No wage records available for analysis."], []
        
        llm_client = self._get_llm_client(mode)
        
        # Prepare data summary for LLM
        # Convert DataFrame to a readable format
        data_summary = df.to_string(index=False, max_rows=50)  # Limit rows for prompt
        if len(df) > 50:
            data_summary += f"\n... (showing first 50 of {len(df)} total rows)"
        
        # Get column statistics
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        stats_summary = []
        for col in numeric_cols[:5]:  # Limit to 5 columns
            if col != "FiscalYear":  # Skip FiscalYear
                stats_summary.append(
                    f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}, "
                    f"mean={df[col].mean():.2f}"
                )
        
        # Group by key dimensions for insights
        insights = []
        if "EmployeeType" in df.columns:
            by_employee = df.groupby("EmployeeType")[numeric_cols[0] if numeric_cols else df.columns[0]].mean().sort_values(ascending=False)
            insights.append(f"Average rates by EmployeeType: {dict(by_employee.head(5))}")
        
        if "SkillType" in df.columns:
            by_skill = df.groupby("SkillType")[numeric_cols[0] if numeric_cols else df.columns[0]].mean().sort_values(ascending=False)
            insights.append(f"Average rates by SkillType: {dict(by_skill.head(5))}")
        
        # Find highest/lowest values
        if numeric_cols:
            for col in numeric_cols[:3]:  # Check first 3 numeric columns
                if col not in ["FiscalYear"]:
                    max_idx = df[col].idxmax()
                    min_idx = df[col].idxmin()
                    max_row = df.loc[max_idx]
                    min_row = df.loc[min_idx]
                    insights.append(f"Highest {col}: {df[col].max():.2f} (EmployeeType={max_row.get('EmployeeType', 'N/A')}, SkillType={max_row.get('SkillType', 'N/A')})")
                    insights.append(f"Lowest {col}: {df[col].min():.2f} (EmployeeType={min_row.get('EmployeeType', 'N/A')}, SkillType={min_row.get('SkillType', 'N/A')})")
        
        prompt = (
            "You are analyzing wage schedule data. Answer the user's question based on the data provided.\n\n"
            f"User Question: {query}\n\n"
            "Data Summary:\n"
            f"{data_summary}\n\n"
            "Key Statistics:\n"
            f"{chr(10).join(stats_summary)}\n\n"
            "Data Insights:\n"
            f"{chr(10).join(insights)}\n\n"
            "IMPORTANT: Return a JSON object with this structure:\n"
            "{\n"
            '  "answer": ["First bullet point answering the question", "Second bullet point", ...],\n'
            '  "tables": [{"headers": ["Column1", "Column2"], "rows": [["Value1", "Value2"]]}]  // OPTIONAL: Use for summary tables\n'
            "}\n\n"
            "RULES:\n"
            "1. Answer the question directly and clearly\n"
            "2. Use specific numbers from the data (min, max, averages)\n"
            "3. Identify which EmployeeType and SkillType have the highest/lowest values if asked\n"
            "4. Compare different categories if the question asks for comparisons\n"
            "5. Each bullet should be at least 10 words long\n"
            "6. If the question asks 'which' or 'what', provide specific answers with values\n"
            "7. Use the 'tables' field for summary comparisons (e.g., top 5 by category)\n"
            "8. Be concise but informative\n"
            "9. Return ONLY valid JSON, no other text\n"
            "10. CRITICAL: When discussing growth rates, trends, or changes over time, use forward-looking language:\n"
            "    - Use 'will increase', 'will decrease', 'is projected to', 'likely to increase', 'expected to grow'\n"
            "    - Instead of past tense like 'increased from', 'was', 'reached', use future/conditional language\n"
            "    - Example: 'The salary will increase from 36.64 in 2022 to 43.84 in 2027' instead of 'increased from'\n"
            "    - Example: 'This represents a growth rate of approximately 19.5% over the five-year period' is fine\n"
            "    - For growth rate questions, frame it as: 'The growth rate will be approximately X%' or 'is expected to be X%'\n"
        )
        
        try:
            self.logger.info(f"[Wage Analysis] Generating LLM analysis for analytical question: {query[:100]}...")
            response = llm_client.generate(prompt, timeout=60)
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group(0))
                answer_points = data.get("answer", [])
                tables = data.get("tables", [])
                
                # Validate answer points
                validated_points = []
                for point in answer_points:
                    if isinstance(point, str) and len(point.strip().split()) >= 10:
                        validated_points.append(point.strip())
                
                # Validate and clean tables
                validated_tables = []
                if isinstance(tables, list):
                    for table in tables:
                        if isinstance(table, dict) and "headers" in table and "rows" in table:
                            if isinstance(table["headers"], list) and isinstance(table["rows"], list):
                                validated_tables.append(table)
                
                if validated_points:
                    self.logger.info(f"[Wage Analysis] Generated {len(validated_points)} answer points and {len(validated_tables)} tables from LLM")
                    return validated_points, validated_tables
                else:
                    self.logger.warning("[Wage Analysis] LLM response had no valid answer points")
                    return [], []
            else:
                self.logger.warning("[Wage Analysis] Could not extract JSON from LLM response")
                return [], []
                
        except Exception as exc:
            self.logger.error(f"[Wage Analysis] Error generating analysis: {exc}", exc_info=True)
            return [], []
    
    def _generate_wage_analysis_tables(self, df: pd.DataFrame, query: str) -> List[Dict[str, Any]]:
        """Generate summary tables for analytical wage questions."""
        tables: List[Dict[str, Any]] = []
        query_lower = query.lower()
        
        # Check if query asks for rankings or comparisons
        is_ranking = any(kw in query_lower for kw in ["which", "highest", "top", "most", "best", "rank"])
        is_comparison = any(kw in query_lower for kw in ["compare", "difference", "different"])
        
        if not is_ranking and not is_comparison:
            return tables
        
        numeric_cols = [col for col in df.columns if df[col].dtype in ("float64", "int64") and col != "FiscalYear"]
        if not numeric_cols:
            return tables
        
        focus_col = numeric_cols[0]  # Use first numeric column (typically a shift column)
        
        # Table 1: Top EmployeeType and SkillType combinations by rate
        if "EmployeeType" in df.columns and "SkillType" in df.columns:
            top_combinations = (
                df.groupby(["EmployeeType", "SkillType"])[focus_col]
                .mean()
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            
            if not top_combinations.empty:
                tables.append({
                    "headers": ["Employee Type", "Skill Type", f"Average {focus_col}"],
                    "rows": [
                        [
                            str(row["EmployeeType"]),
                            str(row["SkillType"]),
                            f"{row[focus_col]:.2f}"
                        ]
                        for _, row in top_combinations.iterrows()
                    ]
                })
        
        # Table 2: By EmployeeType
        if "EmployeeType" in df.columns:
            by_employee = (
                df.groupby("EmployeeType")[focus_col]
                .agg(["mean", "max", "min"])
                .sort_values("mean", ascending=False)
                .reset_index()
            )
            
            if not by_employee.empty:
                tables.append({
                    "headers": ["Employee Type", "Average", "Maximum", "Minimum"],
                    "rows": [
                        [
                            str(row["EmployeeType"]),
                            f"{row['mean']:.2f}",
                            f"{row['max']:.2f}",
                            f"{row['min']:.2f}"
                        ]
                        for _, row in by_employee.iterrows()
                    ]
                })
        
        # Table 3: By SkillType
        if "SkillType" in df.columns:
            by_skill = (
                df.groupby("SkillType")[focus_col]
                .agg(["mean", "max", "min"])
                .sort_values("mean", ascending=False)
                .reset_index()
            )
            
            if not by_skill.empty:
                tables.append({
                    "headers": ["Skill Type", "Average", "Maximum", "Minimum"],
                    "rows": [
                        [
                            str(row["SkillType"]),
                            f"{row['mean']:.2f}",
                            f"{row['max']:.2f}",
                            f"{row['min']:.2f}"
                        ]
                        for _, row in by_skill.iterrows()
                    ]
                })
        
        return tables
    
    def _build_wage_summary_min_max(self, df: pd.DataFrame, question: str) -> List[str]:
        """Build simplified summary with only min/max values and percentage increase if applicable."""
        summary: List[str] = []
        focus_shift = None
        question_lower = question.lower()
        if "shift1" in df.columns:
            if any(term in question_lower for term in ["shift1", "first shift"]):
                focus_shift = "Shift1"
            elif any(term in question_lower for term in ["shift2", "second shift"]):
                focus_shift = "Shift2" if "Shift2" in df.columns else None
            elif any(term in question_lower for term in ["shift3", "third shift"]):
                focus_shift = "Shift3" if "Shift3" in df.columns else None

        numeric_columns = [col for col in df.columns if df[col].dtype in ("float64", "int64")]
        target_columns = [focus_shift] if focus_shift else []
        target_columns += [col for col in ["Shift1", "Shift2", "Shift3"] if col in df.columns and col != focus_shift]
        target_columns = [col for col in target_columns if col in numeric_columns]

        # Mapping for column display names
        col_name_mapping = {
            "Shift1": "Shift1 Hourly Rate",
            "Shift2": "Shift2 Hourly Rate",
            "Shift3": "Shift3 Hourly Rate",
            "Shift1_2_Overtime": "Shift1_2_Overtime Hourly Rate",
            "Shift3_Overtime": "Shift3_Overtime Hourly Rate",
        }
        
        # Check if we have FiscalYear data to calculate percentage increase
        has_fiscal_year = "FiscalYear" in df.columns
        unique_years = sorted(df["FiscalYear"].unique()) if has_fiscal_year else []
        can_calculate_percentage = has_fiscal_year and len(unique_years) >= 2
        
        if target_columns:
            for col in target_columns[:3]:  # Limit to first 3 columns
                col_min = df[col].min()
                col_max = df[col].max()
                # Use renamed column name for consistency with table
                col_name = col_name_mapping.get(col, col)
                
                # Calculate percentage increase if we have fiscal year data spanning multiple years
                percentage_info = ""
                if can_calculate_percentage:
                    # Get values for earliest and latest years
                    earliest_year = unique_years[0]
                    latest_year = unique_years[-1]
                    
                    # Get average value for earliest year
                    earliest_data = df[df["FiscalYear"] == earliest_year][col]
                    latest_data = df[df["FiscalYear"] == latest_year][col]
                    
                    if len(earliest_data) > 0 and len(latest_data) > 0:
                        earliest_value = earliest_data.mean()  # Use mean if multiple rows for same year
                        latest_value = latest_data.mean()
                        
                        if earliest_value > 0:
                            percent_increase = ((latest_value - earliest_value) / earliest_value) * 100
                            percentage_info = f" ({percent_increase:+.1f}% from {earliest_year} to {latest_year})"
                
                summary.append(f"{col_name}: min {col_min:.2f}, max {col_max:.2f}{percentage_info}")
        elif numeric_columns:
            # If no shift columns, use first numeric column
            focus_col = numeric_columns[0]
            col_name = col_name_mapping.get(focus_col, focus_col)
            
            # Calculate percentage increase if applicable
            percentage_info = ""
            if can_calculate_percentage:
                earliest_year = unique_years[0]
                latest_year = unique_years[-1]
                
                earliest_data = df[df["FiscalYear"] == earliest_year][focus_col]
                latest_data = df[df["FiscalYear"] == latest_year][focus_col]
                
                if len(earliest_data) > 0 and len(latest_data) > 0:
                    earliest_value = earliest_data.mean()
                    latest_value = latest_data.mean()
                    
                    if earliest_value > 0:
                        percent_increase = ((latest_value - earliest_value) / earliest_value) * 100
                        percentage_info = f" ({percent_increase:+.1f}% from {earliest_year} to {latest_year})"
            
            summary.append(
                f"{col_name}: min {df[focus_col].min():.2f}, max {df[focus_col].max():.2f}{percentage_info}"
            )

        return summary if summary else []

    @staticmethod
    def _generate_data_summary(df: pd.DataFrame) -> str:
        """Generate a summary of data types and basic statistics."""
        if df.empty:
            return "No data available."
        
        summary_lines = []
        
        # Data types
        summary_lines.append("**Data Types:**")
        for col in df.columns:
            dtype = str(df[col].dtype)
            # Simplify dtype names
            if dtype.startswith("int"):
                dtype_display = "Integer"
            elif dtype.startswith("float"):
                dtype_display = "Float"
            elif dtype == "object":
                dtype_display = "String"
            else:
                dtype_display = dtype
            
            summary_lines.append(f"- {col}: {dtype_display}")
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) > 0:
            summary_lines.append("\n**Numeric Column Statistics:**")
            for col in numeric_cols:
                col_stats = df[col].describe()
                summary_lines.append(
                    f"- {col}: "
                    f"min={col_stats['min']:.2f}, "
                    f"max={col_stats['max']:.2f}, "
                    f"mean={col_stats['mean']:.2f}, "
                    f"median={col_stats['50%']:.2f}"
                )
        
        # Unique value counts for categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns
        if len(categorical_cols) > 0:
            summary_lines.append("\n**Categorical Column Unique Values:**")
            for col in categorical_cols:
                unique_count = df[col].nunique()
                unique_values = df[col].unique()[:5]  # Show first 5 unique values
                if unique_count <= 5:
                    summary_lines.append(f"- {col}: {unique_count} unique values - {', '.join(map(str, unique_values))}")
                else:
                    summary_lines.append(f"- {col}: {unique_count} unique values - {', '.join(map(str, unique_values))} ...")
        
        return "\n".join(summary_lines)

    @staticmethod
    def _prepare_wage_chart(df: pd.DataFrame, query_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare chart data - line chart for multiple years, bar chart for single year."""
        if df.empty:
            return {}
        
        shift_mentioned = query_metadata.get("shift_mentioned", False)
        shift_columns = query_metadata.get("shift_columns", [])
        
        # Determine which columns to chart
        if shift_mentioned and shift_columns:
            # Chart only mentioned shift columns
            columns_to_chart = [col for col in shift_columns if col in df.columns]
        else:
            # Chart all shift columns
            columns_to_chart = [col for col in ["Shift1", "Shift2", "Shift3"] if col in df.columns]
        
        if not columns_to_chart:
            return {}
        
        has_fiscal_year = "FiscalYear" in df.columns
        unique_years = df["FiscalYear"].nunique() if has_fiscal_year else 0
        
        # Determine chart type: line for multiple years, bar for single year or no fiscal year
        chart_type = "line" if unique_years > 1 else "bar"
        
        if has_fiscal_year and unique_years > 1:
            # Line chart: group by FiscalYear
            series_list = []
            for shift_col in columns_to_chart:
                grouped = (
                    df.groupby("FiscalYear")[shift_col]
                    .mean()
                    .round(2)
                    .reset_index()
                )
                points = [
                    {"fiscal_year": int(row["FiscalYear"]), "value": float(row[shift_col])}
                    for _, row in grouped.iterrows()
                ]
                series_list.append({"name": shift_col, "points": points})
            
            return {
                "chart_type": "line",
                "series": series_list,
                "x_axis": "FiscalYear",
                "y_axis": "Wage Rate ($)",
            }
        else:
            # Bar chart: show values across columns
            # If we have fiscal year but only one, group by that
            if has_fiscal_year:
                # Aggregate across all rows (or group by fiscal year if single year)
                chart_data = []
                for shift_col in columns_to_chart:
                    avg_value = df[shift_col].mean()
                    chart_data.append({
                        "name": shift_col,
                        "value": float(round(avg_value, 2))
                    })
            else:
                # No fiscal year - just show averages
                chart_data = []
                for shift_col in columns_to_chart:
                    avg_value = df[shift_col].mean()
                    chart_data.append({
                        "name": shift_col,
                        "value": float(round(avg_value, 2))
                    })
            
            return {
                "chart_type": "bar",
                "series": chart_data,
                "x_axis": "Shift",
                "y_axis": "Wage Rate ($)",
            }

    @staticmethod
    def _normalize_section_slug(value: str) -> str:
        return re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()

    @staticmethod
    def _normalize_disclaimer(text: str) -> str:
        cleaned = re.sub(r"^\s*single[-\s]*sentence\s*reminder\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    @staticmethod
    def _clean_excerpt_for_llm(text: str) -> str:
        """Pre-clean excerpts before sending to LLM to remove artifacts."""
        if not text:
            return ""
        
        # Remove ALL PCCCD.indd patterns (comprehensive removal - must be first)
        # Matches any variation: "PCCCD.indd", "2022 PCCCD.indd", "PCCCD.indd 30", "2022PCCCD.indd", etc.
        text = re.sub(r"\d*\s*PCCCD\.indd\s*\d*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"[Pp][Cc]{2}[Dd]\.indd", "", text)  # Catch any remaining variations
        
        # Remove PDF metadata timestamps (e.g., "2022 PCCCD.indd   142022 PCCCD.indd   14 10/11/24   2:41 PM10/11/24   2:41 PM")
        text = re.sub(r"\d{4}\s+PCCCD\.indd\s+\d+.*?(?:\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*[AP]M)+", "", text)
        
        # Remove standalone timestamps (e.g., "10/11/24   2:41 PM")
        text = re.sub(r"\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*[AP]M", "", text)
        
        # Remove "work 2022 PCCCD.indd" patterns
        text = re.sub(r"work\s+\d{4}\s+PCCCD\.indd", "work", text, flags=re.IGNORECASE)
        
        # Fix hyphenation artifacts (e.g., "combina - tion" -> "combination")
        # Pattern: word part, space, hyphen, space, word continuation
        text = re.sub(r"(\w+)\s+-\s+(\w+)", r"\1\2", text)
        
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        
        return text.strip()

    @staticmethod
    def _clean_answer_point(text: str) -> str:
        """Clean up answer points by removing timestamps, long clause ID lists, and formatting artifacts."""
        if not text:
            return ""
        
        # Remove ALL PCCCD.indd patterns (comprehensive removal - must be first)
        # Matches any variation: "PCCCD.indd", "2022 PCCCD.indd", "PCCCD.indd 30", "2022PCCCD.indd", etc.
        text = re.sub(r"\d*\s*PCCCD\.indd\s*\d*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"[Pp][Cc]{2}[Dd]\.indd", "", text)  # Catch any remaining variations
        
        # Remove PDF metadata timestamps (e.g., "2022 PCCCD.indd   142022 PCCCD.indd   14 10/11/24   2:41 PM10/11/24   2:41 PM")
        text = re.sub(r"\d{4}\s+PCCCD\.indd\s+\d+.*?(?:\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*[AP]M)+", "", text)
        
        # Remove standalone timestamps (e.g., "10/11/24   2:41 PM")
        text = re.sub(r"\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*[AP]M", "", text)
        
        # Remove patterns like "2022 PCCCD.indd 202022 PCCCD.indd 20"
        text = re.sub(r"\d{4}\s+PCCCD\.indd\s+\d+\d{4}\s+PCCCD\.indd\s+\d+", "", text)
        
        # Simplify citations: remove sub-clause IDs and section titles
        # Pattern: "(filename (2.4 (2.4,2.41,2.42,...) – TITLE))" -> "(filename, Section 2.4)"
        # First, handle nested clause ID lists with section titles
        text = re.sub(
            r"\(([^)]+)\s*\(([\d\.]+)\s*\([^)]+\)[^)]*\)\s*–\s*[A-Z\s]+\)\)",
            r"(\1, Section \2)",
            text
        )
        # Handle clause ID lists without nested structure but with section titles
        text = re.sub(
            r"\(([^)]+)\s*\(([\d\.]+),[\d\.,]+\)\s*–\s*[A-Z\s]+\)",
            r"(\1, Section \2)",
            text
        )
        # Handle simple clause ID lists: "(2.4,2.41,2.42,...)" -> "(2.4)"
        text = re.sub(r"\(([\d\.]+),[\d\.,]+\)", r"(\1)", text)
        # Remove "– TITLE" patterns from citations (e.g., "– HOURS AND SHIFTS")
        text = re.sub(r"\(([^)]+)–\s*[A-Z\s]+\)", r"(\1)", text)
        
        # Remove source citations like "(Pacific_Coast_...pdf (6.2...))" or "(Source: ...)"
        # Pattern: "(filename (section) – title)" or "(Source: filename, Section X.Y)"
        # More comprehensive patterns to catch all variations
        
        # Remove citations like "(Pacific-Coast-Clerks-Contract-Document-2022-2028.pdf (8.5 – DISPATCHING, REGISTRATION, AND PREFERENCE))"
        text = re.sub(r"\([^)]*[Pp]acific[^)]*\.pdf[^)]*\)", "", text)
        text = re.sub(r"\([^)]*\.pdf[^)]*\)", "", text)  # Catch any .pdf in parentheses
        text = re.sub(r"\(Source:\s*[^)]+\)", "", text, flags=re.IGNORECASE)
        # Remove nested parentheses patterns like "(filename (section – title))"
        text = re.sub(r"\([^)]*\([^)]*–[^)]*\)[^)]*\)", "", text)
        # Remove patterns with section numbers and titles: "(filename (8.5 – TITLE))"
        text = re.sub(r"\([^)]*\d+\.\d+\s*–\s*[A-Z\s]+\)", "", text)
        # Remove any remaining parenthetical citations that might contain contract references
        text = re.sub(r"\([^)]*[Cc]ontract[^)]*\)", "", text)
        text = re.sub(r"\([^)]*[Cc]lerks[^)]*\)", "", text)
        text = re.sub(r"\([^)]*[Ll]ongshore[^)]*\)", "", text)
        text = re.sub(r"\([^)]*[Ww]alking[^)]*\)", "", text)
        
        # Fix hyphenation artifacts (e.g., "combina - tion" -> "combination")
        # Pattern: word part, space, hyphen, space, word continuation
        text = re.sub(r"(\w+)\s+-\s+(\w+)", r"\1\2", text)
        
        # Remove incomplete sentences that start mid-sentence (common in chunks)
        # Remove sentences that start with lowercase after punctuation
        text = re.sub(r"\.\s+([a-z])", r". \1", text)  # Fix spacing
        
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        
        # Remove trailing closing parentheses that might be left after citation removal
        text = re.sub(r"\s*\)+\s*$", "", text)
        text = re.sub(r"\s+\)\s*$", "", text)
        
        # Remove leading/trailing punctuation artifacts
        text = text.strip(".,;: )")
        
        # Capitalize the first letter of the sentence
        if text:
            # Find the first alphabetic character and capitalize it
            for i, char in enumerate(text):
                if char.isalpha():
                    text = text[:i] + char.upper() + text[i+1:]
                    break
        
        # Remove very short fragments (likely artifacts)
        if len(text.strip()) < 10:
            return ""
        
        return text.strip()

    def _infer_doc_type(self, query: str) -> str:
        """Infer a single document type from query. Returns the first match."""
        doc_types = self._infer_doc_types(query)
        return doc_types[0] if doc_types else "longshore"

    def _infer_doc_types(self, query: str) -> List[str]:
        """
        Infer document types from query. Returns list of types to search.
        Returns empty list or ['longshore', 'clerks', 'walking_bosses'] if none specified.
        """
        lower = query.lower()
        types_found = []
        
        # Check for walking bosses/foremen
        if any(term in lower for term in ["walking boss", "walking bosses", "foremen", "foreman"]):
            types_found.append("walking_bosses")
        
        # Check for clerks
        if "clerk" in lower:
            types_found.append("clerks")
        
        # Check for longshoremen/longshore
        if any(term in lower for term in ["longshore", "longshoremen", "longshoreman"]):
            types_found.append("longshore")
        
        # If nothing found, search all types
        if not types_found:
            return ["longshore", "clerks", "walking_bosses"]
        
        return types_found

    def _craft_opening_text(self, query: str, doc_type: str) -> str:
        doc_label_map = {
            "walking_bosses": "Walking Bosses and Foremen Agreement",
            "clerks": "Clerks Contract Document",
            "longshore": "Longshore Contract Document",
        }
        doc_label = doc_label_map.get(doc_type, "Longshore Contract Document")
        question_summary = re.sub(r"\s+", " ", query.strip())
        return f"Key findings from the {doc_label} regarding “{question_summary}”:"

    def _extract_section_reference(self, query: str) -> Optional[str]:
        """Extract section/clause reference and return just the numeric identifier.
        E.g., 'Section 2' -> '2', 'Clause 2.1' -> '2.1'"""
        for pattern in (SECTION_REF_REGEX, ARTICLE_REF_REGEX, APPENDIX_REF_REGEX, CLAUSE_REF_REGEX):
            match = pattern.search(query)
            if match:
                # Extract the full match (e.g., "section 2" or "clause 2.1")
                full_match = match.group(1)
                # Extract just the numeric part (e.g., "2" or "2.1")
                # Remove the word (section/clause/article/appendix) and keep the number
                numeric_part = re.sub(r'^(?:section|article|appendix|clause)\s+', '', full_match, flags=re.IGNORECASE).strip()
                return numeric_part
        return None

    def _extract_section_id(self, query: str) -> Optional[str]:
        """Extract section_id from query. Looks for patterns like 'section_id: X' or 'section_id=X'."""
        # Pattern for section_id: X or section_id=X
        patterns = [
            re.compile(r"section_id\s*[:=]\s*([A-Za-z0-9\-\.]+)", re.IGNORECASE),
            re.compile(r"section\s+id\s+([A-Za-z0-9\-\.]+)", re.IGNORECASE),
        ]
        for pattern in patterns:
            match = pattern.search(query)
            if match:
                return match.group(1).strip()
        return None

    def _extract_main_clause_id(self, query: str) -> Optional[str]:
        """Extract main_clause_id from query. Looks for patterns like 'main_clause_id: X' or 'clause_id=X'."""
        # Pattern for main_clause_id: X or main_clause_id=X or clause_id: X
        patterns = [
            re.compile(r"main_clause_id\s*[:=]\s*([A-Za-z0-9\-\.]+)", re.IGNORECASE),
            re.compile(r"main\s+clause\s+id\s+([A-Za-z0-9\-\.]+)", re.IGNORECASE),
            re.compile(r"clause_id\s*[:=]\s*([A-Za-z0-9\-\.]+)", re.IGNORECASE),
            re.compile(r"clause\s+id\s+([A-Za-z0-9\-\.]+)", re.IGNORECASE),
        ]
        for pattern in patterns:
            match = pattern.search(query)
            if match:
                return match.group(1).strip()
        return None

    def _extract_section_title(self, query: str) -> Optional[str]:
        """Extract section_title from query. Looks for explicit patterns like 'section_title: "X"' or 'section title: "X"'.
        Only matches when explicitly specified with section_title/section title prefix to avoid false positives."""
        # Pattern for section_title: "X" or section_title='X' or section title: "X"
        # Must have explicit "section_title:" or "section title:" prefix to avoid matching general queries
        patterns = [
            re.compile(r"section_title\s*[:=]\s*['\"]([^'\"]+)['\"]", re.IGNORECASE),
            re.compile(r"section\s+title\s*[:=]\s*['\"]([^'\"]+)['\"]", re.IGNORECASE),
        ]
        for pattern in patterns:
            match = pattern.search(query)
            if match:
                return match.group(1).strip()
        
        # Don't extract general quoted text - only match explicit section_title patterns
        # This prevents false positives from normal queries that happen to have quotes
        return None

    def _build_fallback_summary(self, query: str, sources: List[Dict[str, Any]]) -> List[str]:
        if not sources:
            return []

        keywords = {token.lower() for token in re.findall(r"\b[a-zA-Z]{4,}\b", query)}
        bullets: List[str] = []

        for src in sources:
            excerpt = src.get("excerpt") or ""
            if not excerpt:
                continue

            cleaned = re.sub(r"\s+", " ", excerpt.replace("’", "'")).strip()
            sentences = re.split(r"(?<=[.!?])\s+", cleaned)

            for sentence in sentences:
                sentence_clean = sentence.strip()
                if not sentence_clean:
                    continue

                lower_sentence = sentence_clean.lower()
                if keywords and not any(word in lower_sentence for word in keywords):
                    continue

                label = self._format_source_label(src)
                bullets.append(f"{sentence_clean} ({label})")
                if len(bullets) >= 5:
                    break

            if len(bullets) >= 5:
                break

        if not bullets:
            # Fall back to the first sentences even if keywords didn't match
            for src in sources:
                excerpt = src.get("excerpt") or ""
                if not excerpt:
                    continue
                cleaned = re.sub(r"\s+", " ", excerpt).strip()
                sentence = cleaned.split(". ")[0]
                if sentence:
                    bullets.append(f"{sentence.strip()} ({self._format_source_label(src)})")
                if len(bullets) >= 3:
                    break

        return bullets[:5]

    def _answer_with_generic_llm(self, query: str, mode: str = "llama") -> Dict[str, Any]:
        llm_client = self._get_llm_client(mode)
        api_name = "OpenAI" if mode == "openai" else "Llama"
        self.logger.info(f"[Generic Knowledge] Using {api_name} API to generate response")
        if not llm_client.available():
            # Mode-aware error message
            if mode == "openai":
                error_msg = (
                    "This question appears to be general knowledge. "
                    "Please configure OPENAI_API_KEY environment variable for richer responses."
                )
            else:
                error_msg = (
                    "This question appears to be general knowledge. "
                    "Please configure LLAMA_API_URL for richer responses."
                )
            return {
                "response_type": "generic_llm",
                "content": error_msg,
                "answer_points": [],
                "disclaimer": (
                    "I'm not specialized in general topics, but based on my training data here's a high-level answer. "
                    "For maritime contract questions, please provide specifics."
                ),
                "sources": [],
                "matches": [],
                "total_matches": 0,
            }

        prompt = (
            "Provide a concise, well-structured answer (3-4 bullets) to the user's question. "
            "Do NOT fabricate contract citations. "
            "If the question might need contract-specific nuances, advise consulting the agreements.\n\n"
            f"Question: {query}"
        )
        content = llm_client.generate(prompt, timeout=45, stream=True)
        return {
            "response_type": "generic_llm",
            "content": content,
            "answer_points": [],
            "disclaimer": (
                "I'm not specialized in general topics, but based on my training data here's a high-level answer. "
                "For maritime contract questions, please provide specifics."
            ),
            "sources": [],
            "matches": [],
            "total_matches": 0,
        }

    def _answer_off_topic(self, query: str, mode: str = "llama") -> Dict[str, Any]:
        disclaimer = (
            "I'm tuned for ILWU/PMA maritime agreements. The question seems outside that scope, "
            "but here's a brief response. Please verify details from appropriate sources."
        )

        llm_client = self._get_llm_client(mode)
        api_name = "OpenAI" if mode == "openai" else "Llama"
        self.logger.info(f"[Off-Topic] Using {api_name} API to generate response")
        if not llm_client.available():
            return {
                "response_type": "off_topic",
                "content": disclaimer,
                "matches": [],
                "total_matches": 0,
            }

        prompt = (
            f"{disclaimer}\n\nQuestion: {query}\n\nRespond in 2-3 sentences."
        )
        tail = llm_client.generate(prompt, timeout=30, stream=True)
        return {
            "response_type": "off_topic",
            "content": tail,
            "answer_points": [],
            "disclaimer": disclaimer,
            "sources": [],
            "matches": [],
            "total_matches": 0,
        }

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]

    def _parse_llama_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        json_block = self._extract_json(raw_text)
        if not json_block:
            return None

        try:
            return json.loads(json_block)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([}\]])", r"\1", json_block)
            cleaned = re.sub(r"\\(?![/u\"bfnrt])", "", cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as exc:
                self.logger.warning("Failed to parse LLaMA JSON response: %s", exc)
                return None

