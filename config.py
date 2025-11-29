"""
Configuration module for CQG pipeline.

Centralizes all settings, paths, thresholds, and model configurations.
Uses environment variables for sensitive data (API keys).
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Central configuration for CQG pipeline."""
    
    # ========== Paths ==========
    
    @staticmethod
    def _find_project_root() -> Path:
        """Find project root by looking for marker files."""
        current = Path(__file__).resolve().parent
        for _ in range(4): # Check up to 4 levels up
            if (current / "requirements.txt").exists() or (current / ".git").exists():
                return current
            current = current.parent
        # Fallback to current working directory or script directory
        return Path.cwd()

    PROJECT_ROOT = _find_project_root()
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # Default input/output files
    DEFAULT_INPUT_CSV = DATA_DIR / "transcripts_with_domains.csv"
    DEFAULT_OUTPUT_CSV = OUTPUT_DIR / "task1_queries.csv"
    DEFAULT_METRICS_JSON = OUTPUT_DIR / "task1_metrics.json"
    
    # ========== LLM Configuration ==========
    # Primary LLM for query generation
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))  # seconds
    
    # Judge LLM for validation (cheaper model)
    JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-3-haiku-20240307")
    JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.3"))
    JUDGE_MAX_TOKENS = int(os.getenv("JUDGE_MAX_TOKENS", "500"))
    
    # API Keys (from environment)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Offline controls
    DOMAIN_ONLY_EVENT_DETECTION = os.getenv("DOMAIN_ONLY", "true").lower() == "true"
    DISABLE_LLM = os.getenv("DISABLE_LLM", "auto").lower()  # "true" | "false" | "auto"
    
    # ========== Event Detection ==========
    # SetFit model configuration
    SETFIT_MODEL_PATH = os.getenv(
        "SETFIT_MODEL_PATH",
        "sentence-transformers/paraphrase-mpnet-base-v2"
    )
    
    # Event labels
    EVENT_LABELS: List[str] = [
        "escalation",
        "churn",
        "refund",
        "compliance",
        "general"
    ]
    
    # Event detection thresholds
    EVENT_CONFIDENCE_THRESHOLD = float(os.getenv("EVENT_CONFIDENCE_THRESHOLD", "0.5"))
    
    # Semantic fallback keywords
    EVENT_KEYWORDS = {
        "escalation": ["manager", "supervisor", "escalate", "complaint"],
        "churn": ["cancel", "switch", "competitor", "leave", "unsubscribe"],
        "refund": ["refund", "money back", "return", "reimburse"],
        "compliance": ["privacy", "legal", "gdpr", "violation", "policy"],
        "general": []
    }
    
    # ========== Query Generation ==========
    # Number of queries to generate
    MIN_QUERIES_PER_TRANSCRIPT = int(os.getenv("MIN_QUERIES", "1"))
    MAX_QUERIES_PER_TRANSCRIPT = int(os.getenv("MAX_QUERIES", "3"))
    TARGET_TOTAL_QUERIES = int(os.getenv("TARGET_TOTAL_QUERIES", "50"))
    
    # Query constraints
    MAX_QUERY_WORDS = 50
    REQUIRED_QUESTION_WORDS = ["why", "how", "what"]
    
    # Lehnert taxonomy categories
    LEHNERT_CATEGORIES: List[str] = [
        "antecedent",
        "consequent",
        "goal",
        "enablement"
    ]
    
    # ========== Validation ==========
    # Validation thresholds
    VALIDATION_ACCEPTANCE_THRESHOLD = float(os.getenv("VALIDATION_THRESHOLD", "3.5"))
    MIN_CAUSAL_DEPTH = 2
    MIN_CLARITY = 2
    MIN_BUSINESS_VALUE = 2
    
    # ========== Evidence Grounding ==========
    # Fuzzy matching threshold
    FUZZY_MATCH_THRESHOLD = int(os.getenv("FUZZY_MATCH_THRESHOLD", "85"))
    
    # Stop words for keyword extraction
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'is', 'was', 'were', 'are',
        'did', 'does', 'do', 'why', 'how', 'what', 'when', 'where',
        'who', 'which', 'this', 'that', 'these', 'those', 'of', 'to',
        'in', 'for', 'on', 'with', 'from', 'by', 'at', 'about'
    }
    
    # ========== Logging ==========
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # ========== Performance ==========
    # Caching
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "false").lower() == "true"
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # seconds
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "10"))

    # ========== Clustering ==========
    N_CLUSTERS = 8
    EMBEDDING_MODEL = "sergioburdisso/dialog2flow-joint-dse-base"

    # ========== Characterization ==========
    N_EXEMPLARS = 5
    N_KEY_PHRASES = 5
    SENTIMENT_SAMPLE_SIZE = 20

    # ========== Query Generation ==========
    QUERY_TEMPLATES_ENABLED = [
        "pattern_discovery",
        "outcome_analysis",
        "speaker_dynamics",
        "comparative",
        "training"
    ]
    N_QUERIES_PER_SEED = 3
    DIVERSITY_THRESHOLD = 0.85
    
    @classmethod
    def validate(cls) -> None:
        """Validate required configuration and create directories."""
        # Create directories
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Dynamic defaults: prefer Groq if available and model not explicitly overridden
        if cls.GROQ_API_KEY and os.getenv("LLM_MODEL") is None and cls.LLM_MODEL == "gpt-4o":
            cls.LLM_MODEL = "groq/llama-3.3-70b-versatile"
        if cls.GROQ_API_KEY and os.getenv("JUDGE_MODEL") is None and cls.JUDGE_MODEL == "claude-3-haiku-20240307":
            cls.JUDGE_MODEL = "groq/llama-3.3-70b-versatile"

        has_any_key = any([
            cls.OPENAI_API_KEY, cls.ANTHROPIC_API_KEY,
            cls.GOOGLE_API_KEY, cls.GROQ_API_KEY
        ])

        # Resolve DISABLE_LLM "auto"
        if cls.DISABLE_LLM == "auto":
            cls.DISABLE_LLM = not has_any_key
        else:
            cls.DISABLE_LLM = str(cls.DISABLE_LLM).lower() == "true"

        # Human-friendly notices
        if not has_any_key:
            print("⚠️  Warning: No LLM API keys found – running in offline mode (DISABLE_LLM=true)")
        if cls.DOMAIN_ONLY_EVENT_DETECTION:
            print("ℹ️  Domain-only event detection enabled (no classifier fallback)")
    
    @classmethod
    def get_llm_config(cls) -> dict:
        """Get LLM configuration for litellm."""
        return {
            "model": cls.LLM_MODEL,
            "temperature": cls.LLM_TEMPERATURE,
            "max_tokens": cls.LLM_MAX_TOKENS,
            "timeout": cls.LLM_TIMEOUT,
        }
    
    @classmethod
    def get_judge_config(cls) -> dict:
        """Get judge LLM configuration."""
        return {
            "model": cls.JUDGE_MODEL,
            "temperature": cls.JUDGE_TEMPERATURE,
            "max_tokens": cls.JUDGE_MAX_TOKENS,
            "timeout": cls.LLM_TIMEOUT,
        }


# Initialize and validate config on import
Config.validate()
