import logging
import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import mysql.connector

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
]

WAGE_TABLE_NAME = "wage_schedule_pma"


class Config:
    """Configuration helpers for the contract insights engine."""

    QDRANT_URL = "http://34.131.37.125:6333"
    QDRANT_COLLECTION = "contracts"
    EMBEDDING_API_URL = "http://34.131.37.125:8000/embed"
    EMBEDDING_DIMENSION = 384
    LLAMA_API_URL = "http://34.131.37.125:11434/api/generate"
    LLAMA_MODEL = "llama3"
    LLAMA_TIMEOUT = int(os.getenv("LLAMA_TIMEOUT", "150"))

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
                json={"model": self.model, "prompt": prompt, "stream": True},
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
            json={"model": self.model, "prompt": prompt, "stream": False},
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

        maritime_keywords = [
            "contract",
            "clause",
            "article",
            "pma",
            "ilwu",
            "walking boss",
            "foremen",
            "longshore",
            "dispatcher",
            "agreement",
            "section",
            "grievance",
            "arbitration",
            "vacation",
            "gang",
        ]
        if any(token in query_lower for token in maritime_keywords):
            return "contract_knowledge"

        if any(keyword in query_lower for keyword in WAGE_KEYWORDS):
            return "wage_schedule"

        if not self.llama.available():
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
            if "contract" in label:
                return "contract_knowledge"
            if "off" in label:
                return "off_topic"
        except Exception as exc:  # pragma: no cover
            print(f"[QueryClassifier] Llama classification failed: {exc}")

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
            cursor.close()
            schema_lines = [f"- {col[0]} ({col[1]})" for col in columns]
            self.wage_db_conn = conn
            self.wage_schema = "\n".join(schema_lines)
            self.logger.info("Wage schedule connection established (table=%s)", WAGE_TABLE_NAME)
        except Exception as exc:
            self.logger.warning("Failed to initialize wage schedule connection: %s", exc)
            self.wage_db_conn = None
            self.wage_schema = None

    def handle_query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
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

            section_slug = self._extract_section_reference(query)
            if section_slug:
                section_condition = self.vector_db.make_match_condition("section_slug", section_slug)
                if section_condition:
                    filter_conditions.append(section_condition)

            self.logger.info(
                "Contract query routed | doc_type=%s | section_slug=%s",
                doc_type,
                section_slug or "none",
            )
            debug_info["doc_type"] = doc_type
            if section_slug:
                debug_info["section_slug"] = section_slug

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
                    "section_heading": src.get("section_heading"),
                    "clause": src.get("clause"),
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
            "sources": sources[:3],
            "matches": matches,
            "total_matches": len(matches),
            "opening": opening_text,
            "doc_type": doc_type_for_opening,
        }

    def _build_source_entries(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for match in matches:
            metadata = match.get("metadata") or {}
            page = metadata.get("page") or metadata.get("page_number")
            section_heading = metadata.get("section_heading")
            clause_id = metadata.get("clause")
            clause_heading = metadata.get("clause_heading")
            chunk_type = metadata.get("chunk_type")
            entries.append(
                {
                    "source": match.get("source", "unknown"),
                    "page": page,
                    "score": match.get("score"),
                    "excerpt": match.get("content", "").strip(),
                    "section_heading": section_heading,
                    "clause": clause_id,
                    "clause_heading": clause_heading,
                    "chunk_type": chunk_type,
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
            context_snippets = "\n\n".join(
                f"Source: {self._format_source_label(src)}\nClause: {src.get('excerpt', '')[:800]}"
                for src in sources
            )
            prompt = (
                "You are a contract analyst for ILWU/PMA maritime agreements. "
                "Provide a JSON response with the following schema:\n"
                "{\n"
                '  "answer": ["Concise, plain-language bullet with context and cited clause", "..."],\n'
                '  "disclaimer": "Single-sentence reminder"\n'
                "}\n"
                "Draft 3-6 polished bullets that:\n"
                "- begin with an active verb or clear subject,\n"
                "- plainly summarise the clause impact for the user question,\n"
                "- end with a parenthetical citation of the source (filename page X).\n"
                "Only use the provided clauses. Do NOT invent information.\n\n"
                f"Question: {query}\n\n"
                f"Clauses:\n{context_snippets}"
            )
            try:
                raw_response = self.llama.generate(prompt, stream=True)
                json_payload = self._parse_llama_json(raw_response)
                if json_payload:
                    data = json_payload
                    answer_points = [
                        point.strip() for point in data.get("answer", []) if isinstance(point, str)
                    ]
                    disc = data.get("disclaimer")
                    if isinstance(disc, str) and disc.strip():
                        disclaimer = self._normalize_disclaimer(disc.strip())
            except Exception as exc:  # pragma: no cover
                print(f"[ContractInsightsEngine] Llama synthesis failed: {exc}")
                if len(sources) > 2:
                    try:
                        reduced_snippets = "\n\n".join(
                            f"Source: {self._format_source_label(src)}\nClause: {src.get('excerpt', '')}"
                            for src in sources[:2]
                        )
                        retry_prompt = (
                            "You are a contract analyst for ILWU/PMA maritime agreements. "
                            "Return JSON with keys 'answer' (polished cited bullets) and 'disclaimer'. "
                            "Limit yourself to the supplied clauses.\n\n"
                            f"Question: {query}\n\nClauses:\n{reduced_snippets}"
                        )
                        retry_response = self.llama.generate(
                            retry_prompt,
                            timeout=max(20, Config.LLAMA_TIMEOUT // 2),
                            stream=True,
                        )
                        retry_data = self._parse_llama_json(retry_response)
                        if retry_data:
                            answer_points = [
                                point.strip()
                                for point in retry_data.get("answer", [])
                                if isinstance(point, str)
                            ]
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
        section_heading = source_entry.get("section_heading")
        clause = source_entry.get("clause")
        clause_heading = source_entry.get("clause_heading")

        if clause and clause != "intro":
            label_parts = [clause]
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

    def _generate_wage_sql(self, question: str) -> Optional[str]:
        if not self.llama.available():
            return None

        schema_desc = self.wage_schema or ""
        prompt = (
            "You are an assistant that writes MySQL SELECT queries for the wage schedule table.\n"
            f"Table name: {WAGE_TABLE_NAME}\n"
            f"Columns:\n{schema_desc}\n\n"
            "Rules:\n"
            "- Generate a single SELECT statement.\n"
            "- Infer filters (employee type, skill level, fiscal year, shift) from the question.\n"
            "- Limit the result to 100 rows unless aggregation is required.\n"
            "- Do not include comments or explanations.\n\n"
            f"Question: {question}\n\nSQL:"
        )
        raw_sql = self.llama.generate(prompt, stream=True)
        return self._extract_sql_statement(raw_sql)

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
        sql = sql.split(";")[0]
        return sql.strip()

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
                "opening": f"Wage schedule details for “{query}”",
            }

        sql_query = self._generate_wage_sql(query)
        if not sql_query:
            message = "I couldn't generate a wage schedule query from that question."
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
                "opening": f"Wage schedule details for “{query}”",
            }

        try:
            df = pd.read_sql(sql_query, self.wage_db_conn)
        except Exception as exc:
            self.logger.error("Error executing wage schedule SQL: %s", exc)
            debug_info["wage_sql"] = sql_query
            return {
                "response_type": "wage_schedule_sql",
                "content": f"I encountered an error while running the wage schedule query: {exc}",
                "results": [],
                "sql_query": sql_query,
                "matches": [],
                "sources": [],
                "total_matches": 0,
                "answer_points": [],
                "disclaimer": None,
                "opening": f"Wage schedule details for “{query}”",
            }

        debug_info["wage_sql"] = sql_query
        debug_info["row_count"] = len(df)
        table_text = df.to_string(index=False) if not df.empty else "No rows returned for the requested criteria."

        return {
            "response_type": "wage_schedule_sql",
            "content": f"SQL:\n{sql_query}\n\nResults:\n{table_text}",
            "results": df.to_dict(orient="records"),
            "sql_query": sql_query,
            "matches": [],
            "sources": [],
            "total_matches": len(df),
            "answer_points": [],
            "disclaimer": None,
            "opening": f"Wage schedule details for “{query}”",
        }

    @staticmethod
    def _normalize_section_slug(value: str) -> str:
        return re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()

    @staticmethod
    def _normalize_disclaimer(text: str) -> str:
        cleaned = re.sub(r"^\s*single[-\s]*sentence\s*reminder\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

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

