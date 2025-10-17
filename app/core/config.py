"""
Configuration Module.

Defines dataclasses for managing environment-driven application settings,
including LLM, Qdrant, classifier, Confluence/ServiceNow, and session storage.

Each configuration class loads defaults from environment variables, allowing
flexible deployment across environments without hardcoding values.
"""

import os
from dataclasses import dataclass, field
import json
from typing import Dict, Any
from pathlib import Path

from .core_utils import load_label_mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _load_kwargs(env_var: str) -> Dict[str, Any]:
    """
    Parse additional keyword arguments from an environment variable.

    The variable can be either:
      - A JSON string (preferred).
      - A comma-separated list of key=value pairs.

    Args:
        env_var (str): Name of the environment variable.

    Returns:
        Dict[str, Any]: Parsed key-value pairs (empty if variable is unset).
    """
    raw = os.getenv(env_var, "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        parts = [kv.strip() for kv in raw.split(",") if "=" in kv]
        return {k: v for k, v in (p.split("=", 1) for p in parts)}


@dataclass
class LLMSettings:
    """
    Settings for configuring LLM API client usage.

    Attributes:
        base_url (str): Base URL for the LLM API.
        api_key (str): API key for authentication.
        model (str): Model name to use.
        temperature (float): Sampling temperature for generations.
        max_tokens (int): Maximum number of tokens to generate.
        default_kwargs (Dict[str, Any]): Additional generation parameters,
            parsed from env `LLM_KWARGS`.
    """
    base_url: str = os.getenv("LLM_URL", "https://tritonai-api.ucsd.edu")
    api_key: str = os.getenv("LLM_API_KEY", "")
    model: str = os.getenv("LLM_MODEL", "llama-4-scout")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", 0.2))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", 1024))
    default_kwargs: Dict[str, Any] = field(default_factory=lambda: _load_kwargs("LLM_KWARGS"))


@dataclass
class QdrantSettings:
    """
    Settings for Qdrant vector database connection and retrieval defaults.

    Attributes:
        url (str): Base URL for Qdrant.
        api_key (str): API key for authentication (optional).
        collection_name (str): Name of the Qdrant collection for tickets.
        dense_top_k (int): Default number of results for dense search.
        sparse_top_k (int): Default number of results for sparse search.
    """
    url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key: str = os.getenv("QDRANT_API_KEY")
    collection_name: str = os.getenv("TICKET_COLLECTION", "tickets")
    dense_top_k: int = int(os.getenv("DENSE_TOP_K", "5"))
    sparse_top_k: int = int(os.getenv("SPARSE_TOP_K", "10"))

@dataclass
class ClassifierSettings:
    """
    Settings for classifier model usage, either local or API-based.

    Attributes:
        model_path (str): Path to local model.
        tokenizer_path (str): Path to tokenizer.
        label_mapping_path (str): Path to ID-to-label JSON mapping.
        use_api (bool): Whether to use remote API instead of local model.
        classifier_api (str): API URL for classification.
        classifier_token (str): API token for authentication.
    """
    model_path: str = os.getenv("CLASSIFIER_MODEL_PATH", str(PROJECT_ROOT / "assets/models/itsclassifier/model"))
    tokenizer_path: str = os.getenv("CLASSIFIER_TOKENIZER_PATH", str(PROJECT_ROOT / "assets/models/itsclassifier/tokenizer"))
    label_mapping_path: str = os.getenv(
        "CLASSIFIER_LABEL_MAPPING_PATH",
        str(PROJECT_ROOT / "assets/models/itsclassifier/conversions/id2label.json"),
    )
    use_api: bool = os.getenv("USE_CLASSIFIER_API", "0") == "1"
    classifier_api: str = os.getenv("CLASSIFIER_URL", "https://snowclassifier.ucsd.edu/api/classify")
    classifier_token: str = os.getenv("DSMLP_TOKEN")

    def load_label_mapping(self) -> dict:
        return load_label_mapping(self.label_mapping_path)


@dataclass
class PromptSettings:
    """
    Settings for loading and managing text prompt templates.

    Attributes:
        call_rephrase_prompt_file (str): Path to call rephrase prompt.
        classification_rephrase_prompt_file (str): Path to classification rephrase prompt.
        ticket_rephrase_prompt_file (str): Path to ticket rephrase prompt.
        query_prompt_file (str): Path to retrieval query prompt.
        triage_prompt_file (str): Path to triage prompt.
        troubleshooting_prompt_file (str): Path to troubleshooting prompt.
    """
    call_rephrase_prompt_file: str = os.getenv(
        "CALL_REPHRASE_PROMPT_FILE",
        str(PROJECT_ROOT / "prompts/call_rephrase.txt")
    )
    classification_rephrase_prompt_file: str = os.getenv(
        "CLASSIFICATION_REPHRASE_PROMPT_FILE",
        str(PROJECT_ROOT / "prompts/classification_rephrase.txt")
    )
    ticket_rephrase_prompt_file: str = os.getenv(
        "TICKET_REPHRASE_PROMPT_FILE",
        str(PROJECT_ROOT / "prompts/ticket_rephrase.txt")
    )
    query_prompt_file: str = os.getenv(
        "QUERY_PROMPT_FILE",
        str(PROJECT_ROOT / "prompts/query.txt")
    )
    triage_prompt_file: str = os.getenv(
        "TRIAGE_PROMPT_FILE",
        str(PROJECT_ROOT / "prompts/triage.txt")
    )
    troubleshooting_prompt_file: str = os.getenv(
        "TROUBLESHOOTING_PROMPT_FILE",
        str(PROJECT_ROOT / "prompts/troubleshooting.txt")
    )

    @staticmethod
    def load_prompt(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def get(self, name: str) -> str:
        """
        Get the contents of the prompt file by logical name.
        Example: settings.get("triage") -> contents of triage_prompt_file
        """
        attr_name = f"{name}_prompt_file"
        if not hasattr(self, attr_name):
            raise KeyError(f"No prompt setting found for '{name}'")
        path = getattr(self, attr_name)
        return self.load_prompt(path)

@dataclass
class ConfluenceSettings:
    """
    Settings for Confluence API connection.

    Attributes:
        url (str): Base Confluence URL.
        email (str): User email for authentication.
        api_token (str): API token for authentication.
        space_key (str): Default space key for content retrieval.
    """
    url: str = os.getenv("CONFLUENCE_URL", "https://ucsdcollab.atlassian.net/wiki")
    email: str = os.getenv("CONFLUENCE_EMAIL", "")
    api_token: str = os.getenv("CONFLUENCE_API_KEY", "")
    space_key: str = os.getenv("SPACE_KEY", "CKB")


@dataclass
class ServiceNowGroupsSettings(ConfluenceSettings):
    """
    Settings for ServiceNow Assignment Groups Confluence page.

    Attributes:
        page_title (str): Title of the Confluence page containing groups table.
    """

    page_title: str = os.getenv(
        "PAGE_TITLE", "SNOW Assignment Groups (formerly ITS Business Units)"
    )

@dataclass
class SessionSettings:
    """
    Settings for managing session storage.

    Attributes:
        use_redis (bool): Whether to use Redis for session persistence.
        redis_host (str): Redis host address.
        redis_port (int): Redis port.
        redis_db (int): Redis database index.
        session_prefix (str): Key prefix for session storage.
        session_ttl (int): Session time-to-live (seconds).
    """
    use_redis: bool = os.getenv("USE_REDIS", "0") == "1"
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    session_prefix: str = os.getenv("SESSION_PREFIX", "session:")
    session_ttl: int = int(os.getenv("SESSION_TTL", 3600))  # default 1h