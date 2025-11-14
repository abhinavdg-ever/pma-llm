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
WAGE_KEYWORDS = [
    "wage",
    "salary",
    "pay rate",
    "payroll",
    "shift pay",
    "overtime",
    "hourly",
    "rate of pay",
    "compensation",
    "wages",
    "pay schedule",
    "plot",
    "tabulate",
    "list",
]

WAGE_TABLE_NAME = "wage_schedule_pma"


class Config:
    """Configuration helpers for the contract insights engine."""

    QDRANT_URL = os.getenv("QDRANT_URL", "http://34.131.37.125:6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "contracts")
    EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://34.131.37.125:8000/embed")
    EMBEDDING_DIMENSION = 768  # all-mpnet-base-v2 produces 768-dimensional vectors
    LLAMA_API_URL = os.getenv("LLAMA_API_URL", "http://34.131.37.125:11434/api/generate")
    LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3")
    LLAMA_TIMEOUT = int(os.getenv("LLAMA_TIMEOUT", "150"))
    LLAMA_TEMPERATURE = float(os.getenv("LLAMA_TEMPERATURE", "0.1"))  # Low temperature for more deterministic responses

    @classmethod
    def validate(cls) -> None:
        if not cls.EMBEDDING_API_URL:
            raise ValueError("EMBEDDING_API_URL is not configured.")
        if not cls.QDRANT_URL:
            raise ValueError("QDRANT_URL is not configured.")


class CustomEmbeddingClient:
    """Thin wrapper over the embedding micro-service."""

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

            self.client = QdrantClient(url=qdrant_url)
            self.client.get_collection(collection)  # Raises if missing.
            self.Filter = Filter
            self.FieldCondition = FieldCondition
            self.MatchValue = MatchValue
        except Exception as exc:  # pragma: no cover - graceful degradation.
            print(f"[VectorDatabase] Falling back to mock search: {exc}")
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


class QueryClassifier:
    """Classify questions into contract knowledge, generic, or off-topic buckets."""

    def __init__(self, llama: LlamaClient):
        self.llama = llama

    def classify(self, query: str) -> str:
        query_lower = query.lower()

        # Check wage keywords FIRST (including plot, tabulate, list)
        if any(keyword in query_lower for keyword in WAGE_KEYWORDS):
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
        self.embedder = CustomEmbeddingClient(Config.EMBEDDING_API_URL)
        self.vector_db = VectorDatabase(Config.QDRANT_URL, Config.QDRANT_COLLECTION, self.embedder)
        self.llama = LlamaClient(Config.LLAMA_API_URL, Config.LLAMA_MODEL)
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

    def handle_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        classification = self.classifier.classify(query)
        debug_info: Dict[str, Any] = {
            "query": query,
            "initial_classification": classification,
        }
        self.logger.info("Query received | classification=%s | query=%s", classification, query)

        if classification == "contract_knowledge":
            doc_type = self._infer_doc_type(query)
            filter_conditions: List[Any] = []

            doc_type_condition = self.vector_db.make_match_condition("doc_type", doc_type)
            if doc_type_condition:
                filter_conditions.append(doc_type_condition)

            section_id_filter = self._extract_section_reference(query)
            if section_id_filter:
                section_condition = self.vector_db.make_match_condition("section_id", section_id_filter)
                if section_condition:
                    filter_conditions.append(section_condition)

            self.logger.info(
                "Contract query routed | doc_type=%s | section_id=%s",
                doc_type,
                section_id_filter or "none",
            )
            debug_info["doc_type"] = doc_type
            if section_id_filter:
                debug_info["section_id"] = section_id_filter

            payload = self._answer_with_contract_knowledge(
                query,
                top_k,
                doc_type=doc_type,
                filter_conditions=filter_conditions or None,
                debug_info=debug_info,
            )
        elif classification == "wage_schedule":
            debug_info["doc_type"] = "wage_schedule"
            payload = self._answer_wage_schedule(query, debug_info)
        elif classification == "generic_knowledge":
            payload = self._answer_with_generic_llm(query)
            debug_info["doc_type"] = "n/a"
        else:
            payload = self._answer_off_topic(query)
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

    def _answer_with_contract_knowledge(
        self,
        query: str,
        top_k: int,
        doc_type: Optional[str] = None,
        collection_name: Optional[str] = None,
        filter_conditions: Optional[List[Any]] = None,
        debug_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        matches = self.vector_db.similarity_search(
            query,
            limit=top_k,
            collection_override=collection_name,
            filter_conditions=filter_conditions,
        )
        matches = matches[:top_k]
        if not matches:
            doc_type_label = doc_type or (debug_info.get("doc_type") if debug_info else "longshore")
            if debug_info is not None:
                debug_info["retrieved_sources"] = []
                debug_info["note"] = "No matches from vector search"
            return {
                "response_type": "contract_knowledge",
                "content": "No relevant contract passages were located in the indexed agreements.",
                "answer_points": [],
                "disclaimer": "Consult the official ILWU/PMA agreements for definitive guidance.",
                "sources": [],
                "matches": [],
                "total_matches": 0,
                "opening": self._craft_opening_text(query, doc_type_label),
                "doc_type": doc_type_label,
            }

        sources = self._build_source_entries(matches)
        answer_points, disclaimer = self._synthesize_contract_answer(query, matches, sources)
        content = self._assemble_contract_content(answer_points, disclaimer, sources)

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
                for src in sources[: top_k]
            ]
            self.logger.info(
                "Vector matches | count=%d | details=%s",
                len(debug_info["retrieved_sources"]),
                debug_info["retrieved_sources"],
            )

        doc_type_for_opening = doc_type or (debug_info.get("doc_type") if debug_info else "longshore")
        opening_text = self._craft_opening_text(query, doc_type_for_opening)

        return {
            "response_type": "contract_knowledge",
            "content": content,
            "answer_points": answer_points,
            "disclaimer": disclaimer,
            "sources": sources[:5],
            "matches": matches,
            "total_matches": len(matches),
            "opening": opening_text,
            "doc_type": doc_type_for_opening,
        }

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
    ) -> Tuple[List[str], str]:
        default_disclaimer = "This summary is informational. Refer to the ILWU/PMA agreements for the authoritative language."
        answer_points: List[str] = []
        disclaimer = self._normalize_disclaimer(default_disclaimer)

        if self.llama.available():
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
                '  "disclaimer": "Single sentence reminder"\n'
                "}\n\n"
                "CRITICAL RULES for answer bullets:\n"
                "1. Write in plain, professional language - explain what the contract says in simple terms\n"
                "2. Each bullet should be 1-2 sentences and directly answer the question\n"
                "3. ALWAYS start each bullet with a CAPITALIZED first word\n"
                "4. DO NOT copy raw text from clauses - SUMMARIZE and EXPLAIN the meaning\n"
                "5. ONLY USE SOURCES THAT DIRECTLY ANSWER THE QUESTION - Ignore irrelevant sources\n"
                "   - If asked about holidays, ONLY use holiday-related sources\n"
                "   - If asked about vacations, ONLY use vacation-related sources\n"
                "   - If asked about guarantees, ONLY use guarantee-related sources\n"
                "   - DO NOT include unrelated sections even if they appear in sources\n"
                "6. DO NOT include:\n"
                "   - Timestamps, PDF metadata, long clause ID lists\n"
                "   - Incomplete sentences, raw contract text, or irrelevant content\n"
                "   - Hyphenation artifacts (e.g., fix 'combina - tion' to 'combination')\n"
                "7. End each bullet with a simple citation: (Source: filename, Section X.Y)\n"
                "   - Use ONLY the main clause ID, not sub-clause IDs or long lists\n"
                "8. Focus on WHAT the contract says and WHAT it means for workers\n"
                "9. If the question asks about something specific, ONLY use sources that directly relate to that topic\n\n"
                "Example of GOOD answer bullet:\n"
                '"Clerks are entitled to a 2-hour meal period, and if not sent to eat before the second hour begins, they must be paid for the work performed. (Source: Pacific-Coast-Clerks-Contract-Document-2022-2028.pdf, Section 3.3)"\n\n'
                "Example of BAD answer bullet (DO NOT DO THIS):\n"
                '", the quarter-hour, the half-hour, the three-quarter hour or the even hour and time lost between the designated starting time and time turned to shall be deducted from the guarantee. (Pacific-Coast-Clerks-Contract-Document-2022-2028.pdf (2.4 (2.4,2.41,2.42,2.43,2.44,2.45,2.411,2.431,2.432,2.441,2.442,2.443,2.444,2.445,2.446,2.447,2.448,2.449,2.451,2.452,2.453,2.4441,2.4471,2.4491,2.4492) – HOURS AND SHIFTS))"\n\n'
                "Only use information from the provided clauses. Do NOT invent information.\n\n"
                f"User Question: {query}\n\n"
                f"Relevant Clauses:\n{context_snippets}\n\n"
                "Now provide your JSON response:"
            )
            try:
                raw_response = self.llama.generate(prompt, stream=True)
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
                        if cleaned:
                            answer_points.append(cleaned)
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
                            '  "disclaimer": "Single sentence reminder"\n'
                            "}\n\n"
                            "CRITICAL RULES for answer bullets:\n"
                            "1. Write in plain, professional language - explain what the contract says in simple terms\n"
                            "2. Each bullet should be 1-2 sentences and directly answer the question\n"
                            "3. ALWAYS start each bullet with a CAPITALIZED first word\n"
                            "4. DO NOT copy raw text from clauses - SUMMARIZE and EXPLAIN the meaning\n"
                            "5. ONLY USE SOURCES THAT DIRECTLY ANSWER THE QUESTION - Ignore irrelevant sources\n"
                            "6. DO NOT include:\n"
                            "   - Timestamps, PDF metadata, long clause ID lists\n"
                            "   - Incomplete sentences, raw contract text, or irrelevant content\n"
                            "   - Hyphenation artifacts (e.g., fix 'combina - tion' to 'combination')\n"
                            "7. End each bullet with a simple citation: (Source: filename, Section X.Y)\n"
                            "   - Use ONLY the main clause ID, not sub-clause IDs or long lists\n"
                            "8. Focus on WHAT the contract says and WHAT it means for workers\n"
                            "9. If the question asks about something specific, ONLY use sources that directly relate to that topic\n\n"
                            f"User Question: {query}\n\n"
                            f"Relevant Clauses:\n{retry_context}\n\n"
                            "Now provide your JSON response:"
                        )
                        retry_response = self.llama.generate(
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
                                if cleaned:
                                    answer_points.append(cleaned)
                            disc = retry_data.get("disclaimer")
                            if isinstance(disc, str) and disc.strip():
                                disclaimer = self._normalize_disclaimer(disc.strip())
                    except Exception as retry_exc:  # pragma: no cover
                        print(f"[ContractInsightsEngine] Reduced-context Llama retry failed: {retry_exc}")

        if not answer_points:
            heuristic_points = self._build_fallback_summary(query, sources)
            answer_points.extend(heuristic_points)

        if not answer_points:
            for src in sources[:4]:
                excerpt = src.get("excerpt", "")
                if excerpt:
                    answer_points.append(excerpt.replace("\n", " ").strip()[:240])
        if not answer_points:
            answer_points = ["No synthesized answer available from the retrieved clauses."]

        return answer_points, disclaimer

    def _assemble_contract_content(
        self, answer_points: List[str], disclaimer: str, sources: List[Dict[str, Any]]
    ) -> str:
        answer_section = "\n".join(f"- {point}" for point in answer_points)
        source_section = "\n".join(
            f"- {self._format_source_label(src)}" for src in sources[:5]
        ) or "- No sources available"
        return (
            f"{answer_section}\n\n"
            f"Disclaimer: {disclaimer}\n\n"
            f"Sources:\n{source_section}"
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

    def _generate_wage_sql(self, question: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate SQL and return metadata about the query."""
        if not self.llama.available():
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
        raw_sql = self.llama.generate(prompt, stream=True)
        sql = self._extract_sql_statement(raw_sql)
        return sql, metadata

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

    def _answer_wage_schedule(self, query: str, debug_info: Dict[str, Any]) -> Dict[str, Any]:
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

        sql_query, query_metadata = self._generate_wage_sql(query)
        if not sql_query:
            message = "I couldn't generate a wage schedule query from that question."
            debug_info["wage_sql"] = None
            self.logger.warning("Wage schedule SQL generation failed for query: %s", query)
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
        
        # Generate summary points for answer - just min/max
        if df.empty:
            summary_points: List[str] = ["No wage records matched the filters."]
        else:
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
            "disclaimer": None,
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
        
        # Fix hyphenation artifacts (e.g., "combina - tion" -> "combination")
        # Pattern: word part, space, hyphen, space, word continuation
        text = re.sub(r"(\w+)\s+-\s+(\w+)", r"\1\2", text)
        
        # Remove incomplete sentences that start mid-sentence (common in chunks)
        # Remove sentences that start with lowercase after punctuation
        text = re.sub(r"\.\s+([a-z])", r". \1", text)  # Fix spacing
        
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        
        # Remove leading/trailing punctuation artifacts
        text = text.strip(".,;: ")
        
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
        lower = query.lower()
        if any(term in lower for term in ["walking boss", "walking bosses", "foremen", "foreman"]):
            return "walking_bosses"
        if "clerk" in lower:
            return "clerks"
        return "longshore"

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
        for pattern in (SECTION_REF_REGEX, ARTICLE_REF_REGEX, APPENDIX_REF_REGEX):
            match = pattern.search(query)
            if match:
                return self._normalize_section_slug(match.group(1))
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

    def _answer_with_generic_llm(self, query: str) -> Dict[str, Any]:
        if not self.llama.available():
            return {
                "response_type": "generic_llm",
                "content": (
                    "This question appears to be general knowledge. "
                    "Please configure LLAMA_API_URL for richer responses."
                ),
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
        content = self.llama.generate(prompt, timeout=45, stream=True)
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

    def _answer_off_topic(self, query: str) -> Dict[str, Any]:
        disclaimer = (
            "I'm tuned for ILWU/PMA maritime agreements. The question seems outside that scope, "
            "but here's a brief response. Please verify details from appropriate sources."
        )

        if not self.llama.available():
            return {
                "response_type": "off_topic",
                "content": disclaimer,
                "matches": [],
                "total_matches": 0,
            }

        prompt = (
            f"{disclaimer}\n\nQuestion: {query}\n\nRespond in 2-3 sentences."
        )
        tail = self.llama.generate(prompt, timeout=30, stream=True)
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

