# -*- coding: utf-8 -*-
"""
Sleep Coach LLM System - Demo Mode

This application implements a Sleep Coach LLM system that processes wearable sleep data,
generates personal insights, and provides research-backed guidance using OpenAI models.
The system follows a layered architecture with data processing, analytics, and LLM integration.
"""

import os
import json
import datetime
import mysql.connector
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import uuid
import math
# Math utilities for analytics
class MathUtils:
    @staticmethod
    def mean(values: List[float]) -> float:
        vals = [v for v in values if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    @staticmethod
    def median(values: List[float]) -> float:
        vals = sorted([v for v in values if v is not None])
        n = len(vals)
        if n == 0:
            return 0.0
        mid = n // 2
        if n % 2 == 1:
            return float(vals[mid])
        return (vals[mid - 1] + vals[mid]) / 2.0

    @staticmethod
    def mode(values: List[float]) -> float:
        vals = [v for v in values if v is not None]
        if not vals:
            return 0.0
        freq = {}
        for v in vals:
            freq[v] = freq.get(v, 0) + 1
        return max(freq.items(), key=lambda kv: kv[1])[0]

    @staticmethod
    def percentile(values: List[float], p: float) -> float:
        vals = sorted([v for v in values if v is not None])
        if not vals:
            return 0.0
        p = max(0.0, min(100.0, p))
        if len(vals) == 1:
            return float(vals[0])
        rank = (p / 100.0) * (len(vals) - 1)
        low = int(math.floor(rank))
        high = int(math.ceil(rank))
        if low == high:
            return float(vals[low])
        weight = rank - low
        return float(vals[low] * (1 - weight) + vals[high] * weight)

    @staticmethod
    def summation(values: List[float]) -> float:
        return float(sum([v for v in values if v is not None]))

    @staticmethod
    def minimum(values: List[float]) -> float:
        vals = [v for v in values if v is not None]
        return float(min(vals)) if vals else 0.0

    @staticmethod
    def maximum(values: List[float]) -> float:
        vals = [v for v in values if v is not None]
        return float(max(vals)) if vals else 0.0

# Load environment variables from .env file if available
from dotenv import load_dotenv
load_dotenv()

# Configuration
class Config:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENV")
    PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
    MODEL_NAME = "gpt-4-turbo"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    
    # Local Llama Configuration
    LLAMA_API_URL = os.environ.get("LLAMA_API_URL", "http://34.131.37.125:11434/api/generate")
    LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "llama3")
    
    # Qdrant Vector Database Configuration
    QDRANT_URL = os.environ.get("QDRANT_URL", "http://34.131.37.125:6333")
    QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "docs")
    
    # Custom Embedding Service Configuration
    EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL", "http://34.131.37.125:8000/embed")
    EMBEDDING_DIMENSION = 384  # Based on your embedding service response
    
    # Pinecone Configuration (deprecated, kept for backward compatibility)
    PINECONE_ENDPOINT = os.environ.get("PINECONE_ENDPOINT", "https://aabosleepcoach-xf3zvw5.svc.aped-4627-b74a.pinecone.io")
    
    # Hardcoded User ID (no pseudonymization needed)
    USER_ID = "12"
    
    # MySQL Database Configuration
    DB_CONFIG = {
        'host': os.environ.get('MYSQL_HOST', '62.72.57.99'),
        'user': os.environ.get('MYSQL_USER', 'aabo'),
        'password': os.environ.get('MYSQL_PASSWORD', '3#hxFkBFKJ2Ph!$@'),
        'database': os.environ.get('MYSQL_DATABASE', 'aaboRing10Jan')
    }
    # MySQL tables (per user schema)
    SLEEP_SUMMARY_TABLE = "ai_coach_modules_summary"
    SLEEP_DETAILS_TABLE = "ai_coach_daily_sleep_details"
    
    @classmethod
    def validate(cls):
        # Llama is required, other APIs are optional
        if not cls.LLAMA_API_URL:
            raise ValueError("LLAMA_API_URL not set. Please set LLAMA_API_URL environment variable.")
        print("Configuration validated successfully")

# 1. Raw Sleep Data Processing
@dataclass
class SleepData:
    user_id: str
    date: datetime.date
    sleep_duration: float  # in hours
    deep_sleep: float  # in hours
    rem_sleep: float  # in hours
    light_sleep: float  # in hours
    awake_time: float  # in hours
    heart_rate: List[int]  # beats per minute
    movement: List[float]  # movement intensity
    respiration: List[float]  # breaths per minute
    
    @classmethod
    def from_wearable_data(cls, raw_data: Dict) -> 'SleepData':
        """Process raw wearable data into structured SleepData"""
        # Implementation would depend on the specific wearable device data format
        return cls(
            user_id=raw_data.get('user_id'),
            date=datetime.datetime.fromisoformat(raw_data.get('date')).date(),
            sleep_duration=raw_data.get('sleep_duration', 0),
            deep_sleep=raw_data.get('deep_sleep', 0),
            rem_sleep=raw_data.get('rem_sleep', 0),
            light_sleep=raw_data.get('light_sleep', 0),
            awake_time=raw_data.get('awake_time', 0),
            heart_rate=raw_data.get('heart_rate', []),
            movement=raw_data.get('movement', []),
            respiration=raw_data.get('respiration', [])
        )

# 2. Pseudonymization & PII Redaction
class PrivacyProcessor:
    def __init__(self):
        self.user_id_mapping = {}  # In production, this would be a secure database
    
    def get_pseudo_id(self, user_id: str) -> str:
        """Convert user ID to pseudonymized ID"""
        if user_id not in self.user_id_mapping:
            self.user_id_mapping[user_id] = f"user_{uuid.uuid4().hex[:8]}"
        return self.user_id_mapping[user_id]
    
    def redact_pii(self, data: Dict) -> Dict:
        """Remove all personal identifiers from data"""
        redacted = data.copy()
        if 'user_id' in redacted:
            redacted['user_pseudo_id'] = self.get_pseudo_id(redacted['user_id'])
            del redacted['user_id']
        
        # Remove other PII fields that might be present
        pii_fields = ['name', 'email', 'phone', 'address', 'dob', 'ssn']
        for field in pii_fields:
            if field in redacted:
                del redacted[field]
                
        return redacted

# 3. Time-series Analytics Database
class AnalyticsDatabase:
    def __init__(self):
        self.db_config = Config.DB_CONFIG
        self.summary_table = Config.SLEEP_SUMMARY_TABLE
        self.details_table = Config.SLEEP_DETAILS_TABLE
        self.connection = None
        self.connect_to_db()
        self._summary_columns = None
        self._details_columns = None
        # Resolve actual table names if configured ones are absent
        if self.connection:
            self.summary_table = self._resolve_table(self.summary_table, fallback_contains=["summary"]) or self.summary_table
            self.details_table = self._resolve_table(self.details_table, fallback_contains=["detail", "details"]) or self.details_table
    
    def connect_to_db(self):
        """Connect to MySQL database"""
        try:
            # Increase timeout and add retry logic for better reliability
            connection_params = {
                **self.db_config,
                'connection_timeout': 10,
                'autocommit': True,
                'buffered': True
            }
            self.connection = mysql.connector.connect(**connection_params)
            print("Successfully connected to MySQL database")
        except mysql.connector.Error as err:
            print(f"Error connecting to MySQL database: {err}")
            # Fallback to in-memory storage if database connection fails
            self.daily_summaries = {}

    def _resolve_table(self, preferred_name: str, fallback_contains: List[str]) -> Optional[str]:
        """Return preferred_name if exists; else find first table containing all tokens in fallback_contains."""
        try:
            cursor = self.connection.cursor()
            # Check preferred first
            cursor.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=%s AND table_name=%s",
                (self.db_config.get('database'), preferred_name)
            )
            exists = cursor.fetchone()[0] > 0
            if exists:
                cursor.close()
                return preferred_name
            # Find alternative
            like_clause = " AND ".join(["LOWER(table_name) LIKE %s" for _ in fallback_contains])
            params = tuple([f"%{token.lower()}%" for token in fallback_contains])
            cursor.execute(
                f"SELECT table_name FROM information_schema.tables WHERE table_schema=%s AND {like_clause} ORDER BY table_name ASC",
                (self.db_config.get('database'), *params)
            )
            row = cursor.fetchone()
            cursor.close()
            if row:
                print(f"Resolved table for tokens {fallback_contains}: {row[0]}")
                return row[0]
        except Exception as e:
            print(f"Table resolution error: {e}")
        return None

    def _get_columns(self, table_name: str) -> List[str]:
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=%s",
                (self.db_config.get('database'), table_name)
            )
            cols = [r[0] for r in cursor.fetchall()]
            cursor.close()
            return cols
        except Exception as e:
            print(f"Column introspection error for {table_name}: {e}")
            return []

    def _pick_col(self, candidates: List[str], available: List[str]) -> Optional[str]:
        available_l = {c.lower(): c for c in available}
        for cand in candidates:
            if cand.lower() in available_l:
                return available_l[cand.lower()]
        return None
    
    def store_daily_summary(self, user_id: str, date: datetime.date, metrics: Dict):
        """Store aggregated daily sleep metrics"""
        if not self.connection:
            # Fallback to in-memory storage
            key = f"{user_id}_{date.isoformat()}"
            self.daily_summaries[key] = metrics
            return
        
        # In a real implementation, you would insert/update the database
        # This is a placeholder for demonstration
        print(f"Storing data for user {user_id} on {date}")
    
    def get_user_data(self, user_id: str, start_date: datetime.date, end_date: datetime.date, allow_fallback: bool = True) -> List[Dict]:
        """Retrieve user data for a date range from MySQL using details table columns provided by user.
        If allow_fallback is True and no rows in range, fetch latest recent rows.
        """
        if not self.connection:
            # Fallback to in-memory storage if database connection failed
            results = []
            current_date = start_date
            while current_date <= end_date:
                key = f"{user_id}_{current_date.isoformat()}"
                if hasattr(self, 'daily_summaries') and key in self.daily_summaries:
                    results.append(self.daily_summaries[key])
                current_date += datetime.timedelta(days=1)
            return results
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            # Use exact columns from ai_coach_daily_sleep_details - fetch actual columns dynamically
            details_query = f"""
            SELECT STR_TO_DATE(`date`, '%d/%m/%y') AS date,
                   COALESCE(netDuration, 0) AS netSleepDuration,
                   COALESCE(numAwakenings, 0) AS numAwakenings,
                   COALESCE(awakeDuration, 0) AS awakeDurationMinutes,
                   COALESCE(remDuration, 0) AS remDuration,
                   COALESCE(deepDuration, 0) AS deepDuration,
                   COALESCE(lightDuration, 0) AS lightDuration
            FROM {self.details_table}
            WHERE customer_id = %s
              AND STR_TO_DATE(`date`, '%d/%m/%y') >= %s
              AND STR_TO_DATE(`date`, '%d/%m/%y') <= %s
            ORDER BY STR_TO_DATE(`date`, '%d/%m/%y') ASC
            """
            cursor.execute(details_query, (user_id, start_date, end_date))
            details_rows = cursor.fetchall()
            if details_rows:
                cursor.close()
                print(f"Using details table {self.details_table} with fixed columns: date, netDuration, numAwakenings, awakeDuration")
                return details_rows

            if not allow_fallback:
                cursor.close()
                return []

            # Fallback: fetch latest 60 records regardless of date range
            latest_query = f"""
            SELECT STR_TO_DATE(`date`, '%d/%m/%y') AS date,
                   COALESCE(netDuration, 0) AS netSleepDuration,
                   COALESCE(numAwakenings, 0) AS numAwakenings,
                   COALESCE(awakeDuration, 0) AS awakeDurationMinutes,
                   COALESCE(remDuration, 0) AS remDuration,
                   COALESCE(deepDuration, 0) AS deepDuration,
                   COALESCE(lightDuration, 0) AS lightDuration
            FROM {self.details_table}
            WHERE customer_id = %s
            ORDER BY STR_TO_DATE(`date`, '%d/%m/%y') DESC
            LIMIT 60
            """
            cursor.execute(latest_query, (user_id,))
            latest_rows = cursor.fetchall()
            cursor.close()
            if latest_rows:
                print(f"Falling back to latest records for customer {user_id}: {len(latest_rows)} rows")
                return list(reversed(latest_rows))
            return []
        except mysql.connector.Error as err:
            print(f"Error retrieving data from MySQL: {err}")
            return []

    def customers_overview(self, limit: int = 20) -> List[Dict]:
        """Return recent customers with date ranges for diagnostics."""
        if not self.connection:
            return []
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = f"""
            SELECT customer_id,
                   MIN(`date`) AS first_date,
                   MAX(`date`) AS last_date,
                   COUNT(*) AS records
            FROM {self.details_table}
            GROUP BY customer_id
            ORDER BY last_date DESC
            LIMIT %s
            """
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            cursor.close()
            return rows
        except mysql.connector.Error as err:
            print(f"Error retrieving customers overview: {err}")
            return []
    
    def calculate_trends(self, user_id: str, days: int, allow_fallback: bool = True) -> Dict:
        """Calculate sleep trends over specified number of days"""
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=days)
        print(f"Looking for data for user {user_id} from {start_date} to {today}")
        data = self.get_user_data(user_id, start_date, today, allow_fallback=allow_fallback)
        
        if not data:
            print(f"No data found for user {user_id} in the specified period, trying to get any available data")
            # If no data in the specified period, try to get any available data for this user
            data = self.get_user_data(user_id, datetime.date(2020, 1, 1), today, allow_fallback=allow_fallback)
            if not data:
                overview = self.customers_overview()
                suggestions = ", ".join([str(r.get('customer_id')) for r in overview[:8]]) if overview else "(no suggestions)"
                print(f"No data found for user {user_id} at all. Example customer_ids: {suggestions}")
                return {"error": f"No data for user {user_id}. Try one of: {suggestions}"}
        
        # Process the actual data from database using correct column names
        sleep_durations = []
        awakenings = []
        awake_durations = []
        rem_durations = []
        deep_durations = []
        light_durations = []
        dates = []
        
        for entry in data:
            # Map likely column names from summary/details into expected keys
            date_value = entry.get('date') or entry.get('Date') or entry.get('day')
            if isinstance(date_value, datetime.date):
                dates.append(date_value.isoformat())
            else:
                dates.append(str(date_value))

            # Collect raw values first; we'll normalize units after loop
            sleep_durations.append(entry.get('netSleepDuration', 0) or 0)
            awakenings.append(
                entry.get('numAwakenings') or entry.get('awakenings') or entry.get('awakenings_count') or 0
            )
            awake_durations.append(entry.get('awakeDurationMinutes', 0) or 0)
            rem_durations.append(entry.get('remDuration', 0) or 0)
            deep_durations.append(entry.get('deepDuration', 0) or 0)
            light_durations.append(entry.get('lightDuration', 0) or 0)

        # Heuristic unit normalization:
        # If median per-night sleep value > 24, assume minutes and convert to hours; otherwise assume hours
        def maybe_minutes_to_hours(series: List[float]) -> List[float]:
            med = MathUtils.median(series)
            if med and med > 24:
                return [(v or 0) / 60.0 for v in series]
            return series

        sleep_durations = maybe_minutes_to_hours(sleep_durations)
        rem_durations = maybe_minutes_to_hours(rem_durations)
        deep_durations = maybe_minutes_to_hours(deep_durations)
        light_durations = maybe_minutes_to_hours(light_durations)
        
        # Calculate averages and trends
        trends = {
            "average_sleep_duration": sum(sleep_durations) / len(sleep_durations) if sleep_durations else 0,
            "average_awakenings": sum(awakenings) / len(awakenings) if awakenings else 0,
            "average_awake_duration": sum(awake_durations) / len(awake_durations) if awake_durations else 0,
            "average_rem_duration": sum(rem_durations) / len(rem_durations) if rem_durations else 0,
            "average_deep_duration": sum(deep_durations) / len(deep_durations) if deep_durations else 0,
            "average_light_duration": sum(light_durations) / len(light_durations) if light_durations else 0,
            "sleep_duration_trend": sleep_durations,
            "awakenings_trend": awakenings,
            "awake_duration_trend": awake_durations,
            "rem_duration_trend": rem_durations,
            "deep_duration_trend": deep_durations,
            "light_duration_trend": light_durations,
            "dates": dates,
            "total_records": len(data)
        }
        
        return trends

# 4. Custom Embedding Client
class CustomEmbeddingClient:
    """Client for custom embedding service API"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
        print(f"CustomEmbeddingClient initialized with URL: {self.api_url}")
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text"""
        try:
            import requests
            response = requests.post(
                self.api_url,
                json={"text": text},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                embedding = data.get("embedding", [])
                if not embedding:
                    raise ValueError(f"No embedding returned from API: {data}")
                return embedding
            else:
                raise ValueError(f"Embedding API error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error calling embedding API: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents"""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings

# 5. Local Llama Client
class LlamaClient:
    """Client for interacting with local Llama API"""
    
    def __init__(self):
        self.api_url = Config.LLAMA_API_URL
        self.model = Config.LLAMA_MODEL
        print(f"LlamaClient initialized with URL: {self.api_url}, Model: {self.model}")
    
    def generate(self, prompt: str, timeout: int = 60) -> str:
        """Generate response from local Llama model"""
        try:
            import requests
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True
            }
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=timeout,
                stream=True
            )
            if response.status_code == 200:
                result_text = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            obj = json.loads(line.decode("utf-8"))
                            if "response" in obj:
                                result_text += obj["response"]
                            # Check if done
                            if obj.get("done", False):
                                break
                        except Exception:
                            continue
                return result_text.strip()
            else:
                print(f"Llama API error: {response.status_code} - {response.text}")
                return f"[API ERROR] Trouble connecting to Llama service. Status: {response.status_code}"
        except requests.exceptions.ConnectTimeout as e:
            print(f"Error calling Llama API: Connection timeout - {e}")
            return f"[ERROR] Connection timeout: Unable to connect to Llama API at {self.api_url}. The service may be down or unreachable."
        except requests.exceptions.ConnectionError as e:
            print(f"Error calling Llama API: Connection error - {e}")
            return f"[ERROR] Connection error: Unable to connect to Llama API at {self.api_url}. The service may be down or unreachable."
        except requests.exceptions.Timeout as e:
            print(f"Error calling Llama API: Request timeout - {e}")
            return f"[ERROR] Request timeout: The Llama API took too long to respond."
        except Exception as e:
            print(f"Error calling Llama API: {e}")
            return f"[ERROR] I encountered an error while processing your request: {str(e)}"

# 5. Query Classifier
class QueryClassifier:
    """Uses Llama to classify queries into three categories: SQL, Knowledge, or LLM Training"""
    
    def __init__(self, llama_client: LlamaClient):
        self.llama = llama_client
    
    def is_sleep_related(self, query: str) -> bool:
        """Check if query is related to sleep, health, or wellness before classification"""
        sleep_keywords = [
            'sleep', 'slumber', 'rest', 'nap', 'bedtime', 'wake', 'awake', 'insomnia',
            'dream', 'REM', 'deep sleep', 'light sleep', 'sleep cycle', 'circadian',
            'chronotype', 'fatigue', 'tired', 'energy', 'wellness', 'health',
            'heart rate', 'respiration', 'movement', 'wearable', 'fitness',
            'BMI', 'VO2Max', 'wellness', 'recovery', 'restorative'
        ]
        
        query_lower = query.lower()
        
        # Check for sleep-related keywords
        for keyword in sleep_keywords:
            if keyword in query_lower:
                return True
        
        # Use LLM to check if query is sleep-related if no keywords found
        prompt = f"""Is this query related to sleep, health, wellness, or fitness data?

Query: "{query}"

Answer with ONLY "yes" or "no"."""
        
        try:
            response = self.llama.generate(prompt, timeout=20)
            response_clean = response.strip().upper()
            return "yes" in response_clean or "true" in response_clean
        except Exception as e:
            print(f"Error checking sleep relevance: {e}, defaulting to True")
            return True  # Default to True to avoid false negatives
    
    def classify(self, query: str) -> str:
        """Classify query as 'Data Pull (SQL)', 'Knowledge', 'LLM Core', or 'Off-Topic'"""
        # First check if query is sleep-related
        if not self.is_sleep_related(query):
            print(f"Query '{query}' is not sleep-related, returning 'Off-Topic'")
            return "Off-Topic"
        
        prompt = f"""Classify query into ONE category:

1. Data Pull (SQL): Requests for stored data/metrics (sleep records, trends, chronotype, insomniaScore, DTSScore, BMI, VO2Max, profile data, calculations)
   Examples: "my sleep data", "average sleep", "my chronotype", "insomnia risk", "my BMI", "sleep trends"

2. Knowledge: General sleep facts/science (what is REM sleep, explain sleep cycles, what causes insomnia)
   Examples: "what is REM sleep", "how does deep sleep work", "explain sleep cycles"

3. LLM Core: Advice/recommendations (how to improve sleep, sleep tips, what should I do)
   Examples: "how to improve sleep", "sleep tips", "what should I do for better sleep"

IMPORTANT: Only classify queries related to sleep, health, wellness, or fitness. If the query is about something else (geography, history, general knowledge), return "Off-Topic".

Rules:
- "what is my X" (stored metric) → Data Pull (SQL)
- "what is X" (general sleep concept) → Knowledge  
- "how to/advice/tips" (sleep-related) → LLM Core
- Personal risk scores/metrics → Data Pull (SQL)
- Non-sleep queries → Off-Topic

Query: "{query}"
Response (ONE word only): "Data Pull (SQL)", "Knowledge", "LLM Core", or "Off-Topic"."""
        
        response = self.llama.generate(prompt, timeout=30)
        response_clean = response.strip().upper()
        
        # Extract classification from response
        if "OFF-TOPIC" in response_clean or "OFF TOPIC" in response_clean or "NOT RELATED" in response_clean:
            return "Off-Topic"
        elif "SQL" in response_clean or "DATA PULL" in response_clean:
            return "Data Pull (SQL)"
        elif "KNOWLEDGE" in response_clean:
            return "Knowledge"
        elif "CORE" in response_clean or "LLM" in response_clean:
            return "LLM Core"
        else:
            # Default to Knowledge if unclear (assuming it passed sleep-related check)
            print(f"Unclear classification for query: {query}, response: {response}, defaulting to Knowledge")
            return "Knowledge"

# 6. SQL Agent
class SQLAgent:
    """Uses Llama to generate and execute SQL queries"""
    
    def __init__(self, llama_client: LlamaClient, analytics_db: AnalyticsDatabase):
        self.llama = llama_client
        self.analytics_db = analytics_db
    
    def get_schema_info(self) -> str:
        """Get database schema information for SQL generation - dynamically fetched from database"""
        # Fetch actual columns from database
        details_cols = self.analytics_db._get_columns(self.analytics_db.details_table)
        summary_cols = self.analytics_db._get_columns(self.analytics_db.summary_table)
        
        # Fallback to hardcoded if database query fails
        if not details_cols:
            details_cols = ['customer_id', 'date', 'duration', 'start_time_stamp_converted', 'end_time_stamp_converted', 
                          'nap_flag', 'device_type', 'numAwakenings', 'awakeDuration', 'remDuration', 
                          'lightDuration', 'deepDuration', 'netDuration']
        if not summary_cols:
            summary_cols = ['customer_id', 'dob', 'gender', 'height', 'weight', 'insomniaScore', 
                          'insomniaClassification', 'DTSScore', 'DTSClassification', 'chronotype', 
                          'chronotypeName', 'medianSleepStartTime', 'medianSleepEndTime', 'VO2Max', 'VO2MaxClassification']
        
        schema = f"""Database: {self.analytics_db.db_config['database']}

TABLE 1: {self.analytics_db.details_table} - Daily sleep metrics
Columns: {', '.join(details_cols)}
- customer_id: User identifier
- date: Date in STRING format dd/mm/yy (use STR_TO_DATE(date, '%d/%m/%y'))
- duration: Total sleep duration (minutes)
- start_time_stamp_converted: Sleep start timestamp (dd/mm/yy HH:MM format)
- end_time_stamp_converted: Sleep end timestamp (dd/mm/yy HH:MM format)
- nap_flag: Flag indicating if sleep was a nap (0=no, 1=yes)
- device_type: Type of device used for sleep tracking
- numAwakenings: Number of times user woke up during sleep
- awakeDuration: Time spent awake during sleep (minutes)
- remDuration: REM sleep duration (minutes)
- lightDuration: Light sleep duration (minutes)
- deepDuration: Deep sleep duration (minutes)
- netDuration: Net sleep duration after removing awake time (minutes)

TABLE 2: {self.analytics_db.summary_table} - User profile & risk scores
Columns: {', '.join(summary_cols)}
- customer_id: User identifier
- dob: Date of birth (dd/mm/yy format)
- gender: Gender (Male/Female/Other)
- height: Height in cm
- weight: Weight in kg
- insomniaScore: Insomnia risk score (0-100)
- insomniaClassification: Risk level (No Risk/Mild Risk/Medium Risk/High Risk/Training in Progress)
- DTSScore: Daytime Sleepiness score (0-100)
- DTSClassification: Daytime Sleepiness risk level
- chronotype: Chronotype numeric (0-5)
- chronotypeName: Chronotype name (Morning Types/Evening Types/Biphasic Types/Polyphasic Types/Training in Progress)
- medianSleepStartTime: Median sleep start time (HH:MM:SS)
- medianSleepEndTime: Median sleep end time (HH:MM:SS)
- VO2Max: VO2 Max fitness score (numeric value)
- VO2MaxClassification: VO2 Max classification (Excellent/Good/Average/Fair/Training in Progress)

CRITICAL RULES:
- Date column is VARCHAR in format 'dd/mm/yy' - DO NOT use STR_TO_DATE conversion
- Date comparisons: Convert comparison date to VARCHAR format using DATE_FORMAT (e.g., date >= DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 15 DAY), '%d/%m/%y'))
- Date in SELECT: Use date AS date (no conversion, already readable)
- BMI: Calculate as weight / POWER(height/100, 2) AS bmi (NOT stored)
- GROUP BY: Required when SELECT includes date AND aggregates (AVG/SUM/COUNT/etc). Use: date (no conversion)
- Table aliases: Use consistently (s for summary, d for details) or omit if no JOIN
- JOIN: Both tables use customer_id for relationship

Examples:
- Profile: SELECT chronotypeName, insomniaScore, VO2Max AS vo2_max, weight/POWER(height/100,2) AS bmi FROM {self.analytics_db.summary_table} WHERE customer_id='user123'
- Trends: SELECT date, AVG(netDuration) FROM {self.analytics_db.details_table} WHERE customer_id='user123' GROUP BY date ORDER BY date DESC
- Date filter: SELECT date, AVG(remDuration) AS rem_duration FROM {self.analytics_db.details_table} WHERE customer_id='user123' AND date >= DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 15 DAY), '%d/%m/%y') GROUP BY date ORDER BY date DESC
- Join: SELECT d.netDuration, s.chronotypeName FROM {self.analytics_db.details_table} d JOIN {self.analytics_db.summary_table} s ON d.customer_id=s.customer_id WHERE d.customer_id='user123'"""
        return schema
    
    def generate_sql(self, query: str, customer_id: str) -> str:
        """Generate SQL query from natural language using Llama"""
        schema_info = self.get_schema_info()
        
        prompt = f"""Generate MySQL query for: "{query}" (Customer: {customer_id})

Schema: {schema_info}

Rules:
1. Output ONLY SQL, no explanations
2. Table choice: sleep data/trends → {self.analytics_db.details_table}; profile/risks/chronotype/BMI/VO2Max → {self.analytics_db.summary_table}; both → JOIN
3. Use alias consistently: s.column OR no alias (never mix)
4. Filter: customer_id = '{customer_id}' (unless query asks for all users)
5. Date column is VARCHAR in format 'dd/mm/yy' - DO NOT use STR_TO_DATE conversion
6. Date comparisons in WHERE: Convert comparison date to VARCHAR format using DATE_FORMAT (e.g., date >= DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 15 DAY), '%d/%m/%y'), NOT STR_TO_DATE(date, '%d/%m/%y') >= ...)
7. Date in SELECT: Use date AS date (no conversion needed, it's already in readable format)
8. BMI: Calculate weight/POWER(height/100,2) AS bmi (not stored)
9. GROUP BY: Required when SELECT has date AND aggregates (AVG/SUM/COUNT). Use: date (no conversion)
10. No GROUP BY if only aggregates (no date in SELECT)

SQL:"""
        
        sql_response = self.llama.generate(prompt, timeout=45)
        
        # Check if response is an error message (from Llama API timeout or other errors)
        if (sql_response.startswith("[ERROR]") or 
            sql_response.startswith("[API ERROR]") or 
            "HTTPConnectionPool" in sql_response or 
            "Read timed out" in sql_response or
            "timeout" in sql_response.lower() and "error" in sql_response.lower()):
            error_msg = f"Failed to generate SQL query: {sql_response}"
            print(f"\n❌ {error_msg}")
            raise ValueError(error_msg)
        
        # Extract SQL query from response
        sql_query = sql_response.strip()
        
        # Clean up SQL query - remove markdown code blocks if present
        if "```" in sql_query:
            lines = sql_query.split("\n")
            sql_query = "\n".join([line for line in lines if not line.strip().startswith("```")])
        
        sql_query = sql_query.strip()
        
        # Remove any explanatory text before the SQL query
        # Common patterns: "Here is the SQL query...", "The SQL query is...", etc.
        import re
        
        # Try to find SQL query starting with SELECT
        select_match = re.search(r'(SELECT\s+.*?)(?:;|$)', sql_query, re.IGNORECASE | re.DOTALL)
        if select_match:
            sql_query = select_match.group(1).strip()
            # Remove trailing semicolon if present
            if sql_query.endswith(';'):
                sql_query = sql_query[:-1].strip()
        else:
            # If no SELECT found with regex, try to find first line starting with SELECT
            lines = sql_query.split('\n')
            select_line_idx = None
            for i, line in enumerate(lines):
                if line.strip().upper().startswith('SELECT'):
                    select_line_idx = i
                    break
            
            if select_line_idx is not None:
                # Take everything from the SELECT line onwards
                sql_query = '\n'.join(lines[select_line_idx:]).strip()
                # Remove trailing semicolon if present
                if sql_query.endswith(';'):
                    sql_query = sql_query[:-1].strip()
        
        sql_query = sql_query.strip()
        
        # Basic validation - ensure it's a SELECT query
        if not sql_query.upper().startswith("SELECT"):
            error_msg = f"Failed to generate valid SQL query. Llama returned: {sql_query[:200]}"
            print(f"\n❌ {error_msg}")
            raise ValueError(error_msg)
        
        # Check if query has date and aggregates but missing GROUP BY
        sql_upper = sql_query.upper()
        has_date = "STR_TO_DATE" in sql_query.upper() or "DATE(" in sql_query.upper() or "`date`" in sql_query.lower() or "date AS" in sql_upper
        has_aggregates = any(func in sql_upper for func in ["AVG(", "SUM(", "COUNT(", "MAX(", "MIN(", "ROUND("])
        has_group_by = "GROUP BY" in sql_upper
        
        if has_date and has_aggregates and not has_group_by:
            print(f"\n⚠️  WARNING: Query has date column and aggregates but missing GROUP BY!")
            print(f"Query: {sql_query}")
            print("Attempting to add GROUP BY clause...")
            
            # Try to add GROUP BY before ORDER BY or at the end
            date_expr = "STR_TO_DATE(date, '%d/%m/%y')"
            if "ORDER BY" in sql_upper:
                # Insert GROUP BY before ORDER BY
                sql_query = sql_query.replace("ORDER BY", f"GROUP BY {date_expr}\nORDER BY")
            else:
                # Add GROUP BY at the end
                sql_query = f"{sql_query}\nGROUP BY {date_expr}"
            
            print(f"Fixed query: {sql_query}\n")
        
        return sql_query
    
    def run_query(self, sql_query: str) -> List[Dict]:
        """Run SQL query and get data from database"""
        # Check connection status
        if not self.analytics_db.connection:
            print("\n" + "="*80)
            print("DATABASE CONNECTION ERROR")
            print("="*80)
            print("Database connection not available")
            print("="*80 + "\n")
            return []
        
        # Check if connection is still alive and reconnect if needed
        try:
            self.analytics_db.connection.ping(reconnect=True, attempts=3, delay=1)
        except Exception as ping_error:
            print("\n" + "="*80)
            print("DATABASE CONNECTION ERROR")
            print("="*80)
            print(f"Database connection is not alive: {ping_error}")
            print("Attempting to reconnect...")
            try:
                self.analytics_db.connect_to_db()
                if not self.analytics_db.connection:
                    print("Reconnection failed!")
                    print("="*80 + "\n")
                    return []
                print("Reconnection successful!")
            except Exception as reconnect_error:
                print(f"Reconnection error: {reconnect_error}")
                print("="*80 + "\n")
                return []
        
        try:
            # Basic safety check - only allow SELECT queries
            sql_upper = sql_query.strip().upper()
            if not sql_upper.startswith("SELECT"):
                raise ValueError("Only SELECT queries are allowed")
            
            # Log SQL query
            print("\n" + "="*80)
            print("SQL QUERY EXECUTION")
            print("="*80)
            print(f"SQL Query:\n{sql_query}")
            print("-"*80)
            
            # Execute query with better error handling
            cursor = None
            try:
                cursor = self.analytics_db.connection.cursor(dictionary=True)
                print(f"Cursor created successfully")
                print(f"Executing query...")
                
                cursor.execute(sql_query)
                print(f"Query executed, fetching results...")
                
                results = cursor.fetchall()
                print(f"Results fetched: {len(results)} rows")
                
                cursor.close()
                cursor = None
                
                # Log results
                print(f"Query executed successfully, returned {len(results)} rows")
                if results:
                    print("\nQuery Results (first 10 rows):")
                    print("-"*80)
                    for i, row in enumerate(results[:10], 1):
                        print(f"Row {i}: {row}")
                    if len(results) > 10:
                        print(f"... and {len(results) - 10} more rows")
                    print("-"*80)
                    
                    # Show column names
                    if results:
                        print(f"Columns: {', '.join(results[0].keys())}")
                else:
                    print("No results returned")
                print("="*80 + "\n")
                
                return results
                
            except mysql.connector.Error as db_error:
                # MySQL-specific errors
                if cursor:
                    cursor.close()
                raise db_error
            except Exception as exec_error:
                if cursor:
                    cursor.close()
                raise exec_error
                
        except mysql.connector.Error as e:
            print(f"\n{'='*80}")
            print("SQL QUERY ERROR (MySQL)")
            print("="*80)
            print(f"MySQL Error Code: {e.errno}")
            print(f"MySQL Error Message: {e.msg}")
            print(f"SQL State: {e.sqlstate if hasattr(e, 'sqlstate') else 'N/A'}")
            print(f"Query: {sql_query}")
            import traceback
            print(f"\nTraceback:")
            traceback.print_exc()
            print("="*80 + "\n")
            return []
        except Exception as e:
            print(f"\n{'='*80}")
            print("SQL QUERY ERROR")
            print("="*80)
            print(f"Error Type: {type(e).__name__}")
            print(f"Error executing SQL query: {e}")
            print(f"Query: {sql_query}")
            import traceback
            print(f"\nTraceback:")
            traceback.print_exc()
            print("="*80 + "\n")
            return []
    
    def _serialize_results(self, results: List[Dict]) -> List[Dict]:
        """Convert date/datetime/Decimal objects to JSON-serializable types"""
        from decimal import Decimal
        serialized = []
        for row in results:
            serialized_row = {}
            for key, value in row.items():
                if isinstance(value, (datetime.date, datetime.datetime)):
                    serialized_row[key] = value.isoformat()
                elif isinstance(value, (datetime.timedelta,)):
                    serialized_row[key] = str(value)
                elif isinstance(value, Decimal):
                    # Convert Decimal to float for JSON serialization
                    serialized_row[key] = float(value)
                elif value is None:
                    serialized_row[key] = None
                else:
                    serialized_row[key] = value
            serialized.append(serialized_row)
        return serialized
    
    def format_sql_response(self, query: str, sql_query: str, results: List[Dict]) -> str:
        """Format SQL query results into a user-friendly response using Llama"""
        if not results:
            return "No data found for your query."
        
        # Serialize results to handle date/datetime objects
        serialized_results = self._serialize_results(results[:10])
        
        # Limit results to 10 rows for prompt to avoid token limits
        results_json = json.dumps(serialized_results, indent=2)
        
        prompt = f"""User asked: "{query}"

SQL Results (use ONLY this data, no inventing):
{results_json}

Rules:
- Use exact dates/values from results only
- No invented data
- 5-8 bullet points max
- Be conversational

Response:"""
        
        formatted_response = self.llama.generate(prompt, timeout=45)
        return formatted_response
    
    def process_sql_query(self, query: str, customer_id: str) -> Dict:
        """Process SQL query: generate SQL, execute, and format response (wrapper function)"""
        # Step 1: Generate SQL
        sql_query = self.generate_sql(query, customer_id)
        print(f"Generated SQL: {sql_query}")
        
        # Step 2: Run query and get data
        results = self.run_query(sql_query)
        
        # Step 3: Format response
        if not results:
            return {
                "sql_query": sql_query,
                "results": [],
                "content": "No data found for your query.",
                "total_rows": 0
            }
        
        formatted_content = self.format_sql_response(query, sql_query, results)
        
        return {
            "sql_query": sql_query,
            "results": results,
            "content": formatted_content,
            "total_rows": len(results)
        }

# 7. Cohort Aggregates
class CohortAnalytics:
    def __init__(self, analytics_db: AnalyticsDatabase):
        self.analytics_db = analytics_db
        self.cohort_stats = {
            "sleep_duration": [6.5, 7.0, 7.2, 7.5, 7.8, 8.0, 8.2],
            "deep_sleep": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
            "rem_sleep": [1.5, 1.7, 1.9, 2.1, 2.3, 2.5]
        }
    
    def update_cohort_statistics(self):
        """Update anonymized cohort statistics"""
        print("Demo mode: Mocking cohort statistics update")
    
    def get_percentile(self, metric: str, value: float) -> float:
        """Get percentile of a value within the cohort"""
        if metric not in self.cohort_stats:
            return 50.0  # Default for demo
            
        # Simple percentile calculation for demo
        return 65.0  # Mock percentile

# 8. LangChain Structured Tool Agent
class SleepCoachAgent:
    def __init__(
        self, 
        analytics_db: AnalyticsDatabase,
        cohort_analytics: CohortAnalytics,
        vector_db: Any  # Will be VectorDatabase instance
    ):
        self.analytics_db = analytics_db
        self.cohort_analytics = cohort_analytics
        self.vector_db = vector_db
        
        # Initialize Llama client
        self.llama_client = LlamaClient()
        
        # Initialize query classifier and SQL agent
        self.query_classifier = QueryClassifier(self.llama_client)
        self.sql_agent = SQLAgent(self.llama_client, analytics_db)
        
        # Create LLM wrapper for backward compatibility
        self.llm = type('obj', (object,), {
            'predict': self.llama_client.generate
        })
        
        print(f"Successfully initialized Llama client: {Config.LLAMA_MODEL}")
        
        # Define tools
        self.tools = [
            {
                "name": "AnalyticsTool",
                "func": self.get_personal_data,
                "description": "Fetches personal sleep metrics and trends from Analytics DB"
            },
            {
                "name": "CohortTool",
                "func": self.get_cohort_comparison,
                "description": "Compares user metrics with anonymized cohort data"
            },
            {
                "name": "KnowledgeTool",
                "func": self.get_knowledge,
                "description": "Searches vector DB for sleep-related knowledge and research"
            },
            {
                "name": "ChartTool",
                "func": self.format_chart_data,
                "description": "Formats data for chart generation"
            }
        ]
    
    def get_personal_data(self, query: str, customer_id: str, days: int = 7) -> Dict:
        """Analytics Tool: Get personal sleep data and trends
        Args:
            query: User's query (for context)
            customer_id: Original customer ID for database queries (NOT pseudonymized)
            days: Number of days to analyze
        """
        return self.analytics_db.calculate_trends(customer_id, days)
    
    def get_cohort_comparison(self, metric: str, value: float) -> Dict:
        """Cohort Tool: Compare with anonymized cohort data"""
        percentile = self.cohort_analytics.get_percentile(metric, value)
        return {
            "metric": metric,
            "value": value,
            "percentile": percentile
        }
    
    def get_knowledge(self, query: str) -> List[Dict]:
        """Knowledge Tool: Search vector DB for relevant information"""
        # This would use the vector DB to retrieve relevant documents
        return self.vector_db.similarity_search(query)
    
    def format_chart_data(self, data: Dict) -> Dict:
        """Chart Tool: Format data for chart generation"""
        # Convert data into a format suitable for chart generation
        return {
            "chart_type": "line",  # or other types based on data
            "data": data,
            "format": "json"
        }
    
    def _format_raw_data_summary(self, query: str, results: List[Dict]) -> str:
        """Format raw SQL results into a concise summary without Llama"""
        if not results:
            return "No data found for your query."
        
        lines = []
        
        # Calculate summary statistics
        if results:
            # Find numeric columns (excluding date)
            numeric_cols = []
            date_col = None
            for key in results[0].keys():
                if 'date' in key.lower():
                    date_col = key
                elif isinstance(results[0][key], (int, float)):
                    numeric_cols.append(key)
            
            # Build summary
            if numeric_cols:
                lines.append("Here's a summary of your sleep data:")
                lines.append("")
                
                for col in numeric_cols:
                    values = [r[col] for r in results if r.get(col) is not None]
                    if values:
                        avg_val = sum(values) / len(values)
                        min_val = min(values)
                        max_val = max(values)
                        col_name = col.replace('_', ' ').title()
                        lines.append(f"• {col_name}: Average {avg_val:.1f}, Range {min_val:.1f} - {max_val:.1f}")
        
        return "\n".join(lines)
    
    def process_query(self, customer_id: str, query: str, user_pseudo_id: str = None) -> Dict:
        """Process user query using Llama-based classification
        Args:
            customer_id: Customer ID for database queries (hardcoded to "12")
            query: User's query string
            user_pseudo_id: Ignored - kept for compatibility (defaults to customer_id)
        """
        if user_pseudo_id is None:
            user_pseudo_id = customer_id
        
        response = {
            "query": query,
            "response_type": None,
            "content": None,
            "charts": None,
            "query_classification": None
        }
        
        try:
            # Step 1: Classify query using Llama
            query_type = self.query_classifier.classify(query)
            print(f"Query classified as: {query_type}")
            
            # Store classification in response
            response["query_classification"] = query_type
            
            # Handle off-topic queries - route to LLM Core
            if query_type == "Off-Topic":
                # Route off-topic queries to LLM Core for handling
                query_type = "LLM Core"
                print(f"Off-topic query routed to LLM Core for handling")
            
            if query_type == "Data Pull (SQL)":
                # Step 2a: If SQL, generate SQL query
                print("\n" + "="*80)
                print("SQL QUERY GENERATION")
                print("="*80)
                print(f"User Query: {query}")
                print(f"Customer ID: {customer_id}")
                print("-"*80)
                
                try:
                    sql_query = self.sql_agent.generate_sql(query, customer_id)
                    print(f"Generated SQL: {sql_query}")
                    print("="*80 + "\n")
                except ValueError as e:
                    # SQL generation failed (likely Llama timeout or error)
                    error_msg = str(e)
                    print(f"SQL Generation Error: {error_msg}")
                    print("="*80 + "\n")
                    
                    # Return user-friendly error message
                    response["response_type"] = "error"
                    response["content"] = f"I apologize, but I'm having trouble generating the SQL query right now. This might be due to a timeout or connection issue with the AI service. Please try again in a moment."
                    return response
                
                # Step 2b: Run the query and get data
                results = self.sql_agent.run_query(sql_query)
                
                # Step 2c: Format the response
                # Serialize results to handle date/datetime objects
                serialized_results = self.sql_agent._serialize_results(results) if results else []
                
                if results:
                    # For small datasets (<= 50 rows), use raw data summary (more reliable, no hallucination)
                    # For larger datasets, use Llama for formatting (more conversational)
                    if len(serialized_results) <= 50:
                        formatted_content = self._format_raw_data_summary(query, serialized_results)
                    else:
                        # Try Llama formatting for larger datasets
                        try:
                            formatted_content = self.sql_agent.format_sql_response(query, sql_query, results)
                        except Exception as e:
                            print(f"Error formatting with Llama, using raw data: {e}")
                            formatted_content = self._format_raw_data_summary(query, serialized_results)
                else:
                    formatted_content = "No data found for your query."
                
                response["response_type"] = "sql_query"
                response["content"] = formatted_content
                response["sql_query"] = sql_query
                # Always include raw results so client can use them directly
                response["results"] = serialized_results
                response["total_rows"] = len(results) if results else 0
                
            elif query_type == "Knowledge":
                # Step 2b: If Knowledge, first search Qdrant knowledge base
                knowledge_results = self.get_knowledge(query)
                
                # Step 3: Generate response using Llama
                if knowledge_results and len(knowledge_results) > 0:
                    # Found relevant knowledge - use it as context with source citations
                    knowledge_context = "\n".join([
                        f"- {result.get('content', '')} (Source: {result.get('source', 'Unknown')})" 
                        for result in knowledge_results[:3]
                    ])
                    sources = [result.get('source', 'Unknown') for result in knowledge_results[:3]]
                    
                    prompt = f"""Answer: "{query}"

Knowledge: {knowledge_context}

Rules:
- 5-8 bullet points (start with "- ")
- Use knowledge provided
- Cite sources
- Suggest healthcare provider if needed
- Be concise

Response:"""
                    
                    # Try to generate response with Llama, with timeout handling
                    try:
                        llm_response = self.llama_client.generate(prompt, timeout=60)
                        
                        # Append sources to response
                        if sources:
                            sources_text = "\n\n**Sources:**\n" + "\n".join([f"- {source}" for source in set(sources)])
                            llm_response += sources_text
                        
                        response["response_type"] = "knowledge_query"
                        response["content"] = llm_response
                        response["knowledge_sources"] = sources
                        response["knowledge_used"] = True
                    except Exception as e:
                        # Timeout or other error - use knowledge directly as fallback
                        error_msg = str(e)
                        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                            print(f"⚠️  Llama API timeout during knowledge query. Using knowledge sources directly.")
                            # Format knowledge sources as response
                            fallback_response = f"Based on the available knowledge sources:\n\n"
                            for i, result in enumerate(knowledge_results[:3], 1):
                                content = result.get('content', '')
                                source = result.get('source', 'Unknown')
                                fallback_response += f"{i}. {content}\n"
                            
                            if sources:
                                sources_text = "\n\n**Sources:**\n" + "\n".join([f"- {source}" for source in set(sources)])
                                fallback_response += sources_text
                            
                            response["response_type"] = "knowledge_query"
                            response["content"] = fallback_response
                            response["knowledge_sources"] = sources
                            response["knowledge_used"] = True
                        else:
                            # Other error - re-raise or handle differently
                            raise
                else:
                    # No knowledge found in Qdrant - fall back to LLM Core
                    prompt = f"""Answer: "{query}"

Provide general sleep advice:
- 5-8 bullet points (start with "- ")
- Based on sleep science
- Suggest healthcare provider if needed
- Be concise

Response:"""
                    
                    try:
                        llm_response = self.llama_client.generate(prompt, timeout=60)
                        response["response_type"] = "llm_core_query"
                        response["content"] = llm_response
                        response["knowledge_used"] = False
                        response["knowledge_sources"] = []
                    except Exception as e:
                        error_msg = str(e)
                        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                            print(f"⚠️  Llama API timeout during LLM Core query.")
                            response["response_type"] = "llm_core_query"
                            response["content"] = "I apologize, but the AI service is taking longer than expected to respond. Please try again in a moment, or rephrase your question."
                            response["knowledge_used"] = False
                            response["knowledge_sources"] = []
                        else:
                            raise
            
            elif query_type == "LLM Core":
                # Step 2b: For LLM Core queries, use LLM training directly
                # Check if this was originally an off-topic query
                is_off_topic = response["query_classification"] == "Off-Topic"
                
                if is_off_topic:
                    # Handle off-topic queries with a friendly redirect
                    prompt = f"""You are a Sleep Coach AI assistant. A user asked: "{query}"

This question is not related to sleep, health, or wellness. Please:
1. Politely acknowledge that you're a Sleep Coach AI specializing in sleep-related topics
2. Briefly answer the question if possible
3. Suggest how you can help with sleep-related questions

Be friendly and helpful. Keep response concise (3-5 sentences).

Response:"""
                else:
                    # Regular sleep-related LLM Core query
                    prompt = f"""Answer: "{query}"

Provide personalized sleep advice:
- 5-8 bullet points (start with "- ")
- Practical, actionable
- Based on sleep science
- Suggest healthcare provider if needed
- Be concise

Response:"""
                
                try:
                    llm_response = self.llama_client.generate(prompt, timeout=60)
                    # Check if the response is an error message
                    if llm_response.startswith("[ERROR]") or llm_response.startswith("[API ERROR]"):
                        print(f"⚠️  Llama API error returned: {llm_response}")
                        if is_off_topic:
                            response["content"] = "I'm a Sleep Coach AI assistant specializing in sleep-related questions. While I'd like to help with your question, I'm experiencing some technical delays. Please try asking me something about sleep, health, or wellness!"
                        else:
                            response["content"] = "I apologize, but the AI service is taking longer than expected to respond. Please try again in a moment, or rephrase your question."
                        response["response_type"] = "llm_core_query"
                        response["knowledge_used"] = False
                        response["knowledge_sources"] = []
                    else:
                        response["response_type"] = "llm_core_query"
                        response["content"] = llm_response
                        response["knowledge_used"] = False
                        response["knowledge_sources"] = []
                except Exception as e:
                    error_msg = str(e)
                    if ("timeout" in error_msg.lower() or "timed out" in error_msg.lower() or 
                        "ConnectTimeout" in error_msg or "ConnectionError" in error_msg or
                        "Max retries exceeded" in error_msg):
                        print(f"⚠️  Llama API connection/timeout error during LLM Core query: {error_msg}")
                        if is_off_topic:
                            response["content"] = "I'm a Sleep Coach AI assistant specializing in sleep-related questions. While I'd like to help with your question, I'm experiencing some technical delays. Please try asking me something about sleep, health, or wellness!"
                        else:
                            response["content"] = "I apologize, but the AI service is taking longer than expected to respond. Please try again in a moment, or rephrase your question."
                        response["response_type"] = "llm_core_query"
                        response["knowledge_used"] = False
                        response["knowledge_sources"] = []
                    else:
                        raise
                
        except Exception as e:
            print(f"Error processing query: {e}")
            import traceback
            traceback.print_exc()
            response["response_type"] = "error"
            response["content"] = f"I encountered an error while processing your request: {str(e)}. Please try again."
            
        return response

# 9. Vector DB (Qdrant)
class VectorDatabase:
    def __init__(self):
        try:
            # Try to import Qdrant client
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams, PointStruct
                USE_QDRANT = True
            except ImportError:
                USE_QDRANT = False
            
            if not USE_QDRANT:
                raise ImportError("Qdrant client not available")
            
            # Initialize custom embedding client
            self.embeddings = CustomEmbeddingClient(Config.EMBEDDING_API_URL)
            
            # Initialize Qdrant client
            self.qdrant_url = Config.QDRANT_URL
            self.collection_name = Config.QDRANT_COLLECTION_NAME
            self.client = QdrantClient(url=self.qdrant_url)
            
            # Check if collection exists
            try:
                collection_info = self.client.get_collection(self.collection_name)
                print(f"Collection '{self.collection_name}' exists with {collection_info.points_count} points")
            except Exception:
                # Collection doesn't exist, but we'll use it anyway (assume it exists on server)
                print(f"Note: Collection '{self.collection_name}' check failed, but continuing...")
                print(f"Collection may need to be created on Qdrant server with:")
                print(f"  - dimension: {Config.EMBEDDING_DIMENSION}")
                print(f"  - distance: Cosine")
            
            self.vectorstore = None  # We'll use direct Qdrant API
            print(f"Successfully connected to Qdrant at {self.qdrant_url}")
            print(f"Using collection: {self.collection_name}")
            
        except Exception as e:
            # Fallback to mock implementation
            if isinstance(e, ImportError):
                print(f"⚠️  Qdrant client not installed. Install with: pip install qdrant-client")
                print("   Falling back to mock Vector Database (using default knowledge)")
            else:
                print(f"⚠️  Error initializing Vector Database: {e}")
                print("   Falling back to mock Vector Database (using default knowledge)")
            self.embeddings = None
            self.client = None
            self.vectorstore = None
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into chunks with overlap for better context preservation"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + chunk_size
            
            # If not the last chunk, try to break at sentence/paragraph boundary
            if end < len(text):
                # Try to find a good break point (sentence end, paragraph, or newline)
                for break_char in ['\n\n', '\n', '. ', '! ', '? ']:
                    last_break = text.rfind(break_char, start, end)
                    if last_break != -1:
                        end = last_break + len(break_char)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def add_documents(self, documents: List[Dict], chunk_size: int = 1000, chunk_overlap: int = 200):
        """Add documents to the vector database with automatic chunking
        
        Args:
            documents: List of dicts with 'content' and 'source' keys
            chunk_size: Maximum characters per chunk (default: 1000)
            chunk_overlap: Characters to overlap between chunks (default: 200)
        """
        if not self.client or not self.embeddings:
            print(f"Mock: Adding {len(documents)} documents to vector database")
            return
            
        try:
            from qdrant_client.models import PointStruct
            import uuid
            
            # Chunk all documents
            all_chunks = []
            for doc in documents:
                content = doc.get('content', '')
                source = doc.get('source', 'unknown')
                
                # Chunk the content
                chunks = self._chunk_text(content, chunk_size, chunk_overlap)
                
                for chunk_idx, chunk in enumerate(chunks):
                    all_chunks.append({
                        'content': chunk,
                        'source': source,
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks)
                    })
            
            print(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")
            
            # Generate embeddings for all chunks
            texts = [chunk['content'] for chunk in all_chunks]
            embeddings = self.embeddings.embed_documents(texts)
            
            # Prepare points for Qdrant
            points = []
            for chunk, embedding in zip(all_chunks, embeddings):
                point_id = str(uuid.uuid4())
                payload = {
                    "text": chunk['content'],
                    "source": chunk['source'],
                    "chunk_index": chunk['chunk_index'],
                    "total_chunks": chunk['total_chunks']
                }
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            # Upsert points to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Added {len(all_chunks)} chunks from {len(documents)} documents to Qdrant collection '{self.collection_name}'")
        except Exception as e:
            error_msg = str(e)
            print(f"Error adding documents to Qdrant: {error_msg}")
            import traceback
            traceback.print_exc()
    
    def similarity_search(self, query: str, k: int = 3):
        """Search for similar documents"""
        if not self.client or not self.embeddings:
            # Return mock results if Qdrant not available
            return [
                {
                    'content': "Adults need 7-9 hours of sleep per night for optimal health.",
                    'source': "National Sleep Foundation"
                },
                {
                    'content': "Deep sleep is crucial for physical recovery and immune function.",
                    'source': "Journal of Sleep Research"
                },
                {
                    'content': "REM sleep plays a vital role in memory consolidation and emotional processing.",
                    'source': "Neuroscience & Biobehavioral Reviews"
                }
            ]
            
        try:
            # Generate embedding for query
            query_embedding = self.embeddings.embed_query(query)
            
            # Perform similarity search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k
            )
            
            # Format results
            formatted_results = []
            for result in search_results:
                payload = result.payload or {}
                formatted_results.append({
                    'content': payload.get('text', ''),
                    'source': payload.get('source', 'unknown'),
                    'score': result.score
                })
                
            return formatted_results
        except Exception as e:
            print(f"Error performing similarity search: {e}")
            import traceback
            traceback.print_exc()
            # Return mock results as fallback
            return [
                {
                    'content': "Adults need 7-9 hours of sleep per night for optimal health.",
                    'source': "National Sleep Foundation"
                },
                {
                    'content': "Deep sleep is crucial for physical recovery and immune function.",
                    'source': "Journal of Sleep Research"
                },
                {
                    'content': "REM sleep plays a vital role in memory consolidation and emotional processing.",
                    'source': "Neuroscience & Biobehavioral Reviews"
                }
            ]

# 10. Conversation & Logging
class ConversationLogger:
    def __init__(self):
        self.sessions = {}  # In production, this would be a database
    
    def log_interaction(self, user_id: str, query: str, response: Dict):
        """Log user interaction"""
        session_id = f"session_{user_id}_{datetime.datetime.now().date().isoformat()}"
        
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "response": response
        })
    
    def get_session_history(self, user_id: str) -> List[Dict]:
        """Get session history for a user"""
        session_id = f"session_{user_id}_{datetime.datetime.now().date().isoformat()}"
        return self.sessions.get(session_id, [])

# Main Sleep Coach LLM Application
class SleepCoachLLM:
    def __init__(self):
        # Initialize components (no privacy processor needed)
        self.analytics_db = AnalyticsDatabase()
        self.cohort_analytics = CohortAnalytics(self.analytics_db)
        self.vector_db = VectorDatabase()
        self.agent = SleepCoachAgent(
            self.analytics_db,
            self.cohort_analytics,
            self.vector_db
        )
        self.logger = ConversationLogger()
        # Hardcoded user ID
        self.user_id = Config.USER_ID
        
    def process_wearable_data(self, raw_data: Dict) -> None:
        """Process incoming wearable data"""
        # 1. Convert to structured format
        sleep_data = SleepData.from_wearable_data(raw_data)
        
        # 2. Store in analytics database (using hardcoded user_id)
        metrics = {
            "date": sleep_data.date.isoformat(),
            "sleep_duration": sleep_data.sleep_duration,
            "deep_sleep": sleep_data.deep_sleep,
            "rem_sleep": sleep_data.rem_sleep,
            "light_sleep": sleep_data.light_sleep,
            "awake_time": sleep_data.awake_time,
            "average_heart_rate": sum(sleep_data.heart_rate) / len(sleep_data.heart_rate) if sleep_data.heart_rate else 0,
            "average_respiration": sum(sleep_data.respiration) / len(sleep_data.respiration) if sleep_data.respiration else 0,
            "movement_index": sum(sleep_data.movement) / len(sleep_data.movement) if sleep_data.movement else 0
        }
        
        self.analytics_db.store_daily_summary(self.user_id, sleep_data.date, metrics)
        
        # 3. Update cohort statistics
        self.cohort_analytics.update_cohort_statistics()
        
    def handle_user_query(self, user_id: str, query: str) -> Dict:
        """Handle user query about sleep data or knowledge
        Args:
            user_id: Ignored - using hardcoded user_id from Config
            query: User's query string
        """
        # Use hardcoded user_id from Config
        customer_id = self.user_id
        
        # Process query through agent (using hardcoded customer_id)
        response = self.agent.process_query(customer_id, query, customer_id)
        
        # Log interaction (using hardcoded user_id)
        self.logger.log_interaction(customer_id, query, response)
        
        return response
    
    def add_knowledge_documents(self, documents: List[Dict]) -> None:
        """Add sleep-related documents to knowledge base"""
        self.vector_db.add_documents(documents)

# Example usage
def main():
    print("\n===== Sleep Coach LLM Demo =====\n")
    
    # Initialize Sleep Coach LLM
    sleep_coach = SleepCoachLLM()
    
    # Example: Add knowledge documents
    print("\n1. Adding knowledge documents to vector database...")
    sleep_coach.add_knowledge_documents([
        {
            "content": "Research shows that adults need 7-9 hours of sleep per night for optimal health.",
            "source": "National Sleep Foundation"
        },
        {
            "content": "Deep sleep is crucial for physical recovery and immune function.",
            "source": "Journal of Sleep Research"
        },
        {
            "content": "REM sleep plays a vital role in memory consolidation and emotional processing.",
            "source": "Neuroscience & Biobehavioral Reviews"
        }
    ])
    
    # Example: Process wearable data
    print("\n2. Processing sample wearable sleep data...")
    sleep_coach.process_wearable_data({
        "user_id": "user123",
        "date": "2023-06-01",
        "sleep_duration": 7.5,
        "deep_sleep": 1.2,
        "rem_sleep": 1.8,
        "light_sleep": 4.0,
        "awake_time": 0.5,
        "heart_rate": [60, 58, 62, 57, 59],
        "movement": [0.1, 0.2, 0.1, 0.3, 0.1],
        "respiration": [14, 15, 14, 13, 15]
    })
    
    # Example: Handle user queries
    print("\n3. Testing different types of user queries:")
    
    # Personal data query
    print("\n3.1 Personal data query:")
    personal_query = "How was my sleep over the last 7 days?"
    print(f"User query: '{personal_query}'")
    response = sleep_coach.handle_user_query("user123", personal_query)
    print(f"Response type: {response['response_type']}")
    print(f"Content: {response['content']}")
    
    # Cohort comparison query
    print("\n3.2 Cohort comparison query:")
    cohort_query = "How does my deep sleep compare to others?"
    print(f"User query: '{cohort_query}'")
    response = sleep_coach.handle_user_query("user123", cohort_query)
    print(f"Response type: {response['response_type']}")
    print(f"Content: {response['content']}")
    
    # Knowledge-based query
    print("\n3.3 Knowledge-based query:")
    knowledge_query = "What's the importance of REM sleep?"
    print(f"User query: '{knowledge_query}'")
    response = sleep_coach.handle_user_query("user123", knowledge_query)
    print(f"Response type: {response['response_type']}")
    print(f"Content: {response['content']}")
    
    print("\n===== Demo Complete =====")

if __name__ == "__main__":
    # Check if running as script or imported as module
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        # Running as web app, don't execute demo
        pass
    else:
        main()