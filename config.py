import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE, override=False)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")

"""
    Convenience function for boolean values 
"""
def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default

    normalized = value.strip().lower()

    if normalized in {"1", "true", "yes", "y", "on"}:
        return True

    if normalized in {"0", "false", "no", "n", "off"}:
        return False 

    raise ValueError(f"Invalid boolean configuration value: {value!r}")
 

def _project_path(value: str) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


@dataclass(frozen=True)
class Settings:
    # RAG settings
    rag_auth_enabled: bool = _as_bool (
        str(os.getenv("RAG_AUTH_ENABLED", "true").lower())
    )
    api_key: str = os.getenv("RAG_API_KEY", "").strip()

    # Document store settings
    store_backend: str = os.getenv("STORE_BACKEND", "in_memory")
    documents_root: Path = _project_path(
        os.getenv("DOCUMENTS_ROOT", "./documents")
    )

    # In Memory store settings
    in_memory_persist: bool = _as_bool(os.getenv("IN_MEMORY_PERSIST","true").lower())
    in_memory_store_path: str = _project_path( 
        os.getenv(
            "IN_MEMORY_STORE_PATH",
            ".data/in-memory-document-store.json"
        )
    )

    # Open search settings
    opensearch_url: str = os.getenv(
        "OPENSEARCH_URL", "https://localhost:9200"
    )
    opensearch_index: str = os.getenv("OPENSEARCH_INDEX", "rag-documents")
    opensearch_user: str = os.getenv("OPENSEARCH_USERNAME", "")
    opensearch_password: str = os.getenv("OPENSEARCH_PASSWORD", "")
    opensearch_verify_certs: bool = _as_bool(
        str(os.getenv("OPENSEARCH_VERIFY_CERTS", "true").lower())  
    )
    opensearch_ca_certs: str = os.getenv("OPENSEARCH_CA_CERTS", "")
    opensearch_timeout: int = os.getenv(
        "OPENSEARCH_TIMEOUT",
        "60"
    )

    # Embedding model settings
    embedding_model: str = str(
        _project_path(
            os.getenv(
                "EMBEDDING_MODEL",
                "/opt/models/all-mpnet-base-v2",
            )
        )
    )
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "768"))
    embedding_batch_size: int = int(
        os.getenv(
            "EMBEDDING_BATCH_SIZE",
            "16"
        )
    )
    embedding_top_k: int = int(
        os.getenv(
            "EMBEDDING_TOP_K",
            "20"
        )
    )
    normalize_embeddings: bool = _as_bool (
        str(os.getenv("NORMALIZE_EMBEDDINGS","true").lower())
    )
    chunk_size: int = int(
        os.getenv(
            "CHUNK_SIZE",
            "20"
        )
    )
    chunk_overlap: int = int(
        os.getenv(
            "CHUNK_OVERLAP",
            "40"
        )
    )
    bm25_top_k: int = int( 
        os.getenv(
            "BM25_TOP_K",
            "20"
        )
    )
    # Reranker model settings
    reranker_enabled: bool = _as_bool (
        str(os.getenv("RERANKER_ENABLED", "true").lower())
    )
    reranker_model: str = str(
        _project_path(
            os.getenv(
                "RERANKER_MODEL",
                "/opt/models/ms-marco-MiniLM-L-6-v2",
            )
        )
    )
    reranker_top_k: int = int(
        os.getenv(
            "RERANK_TOP_K",
            "20"
        )
    )

    # Inference server settings
    llm_base_url: str = os.getenv(
        "LLM_BASE_URL", "http://localhost:8033/v1"
    )
    llm_model: str = os.getenv("LLM_MODEL", "local-chat")
    llm_api_key: str = os.getenv("LLM_API_KEY", "not-needed")

    @property
    def document_store_backend(self) -> str:
        return self.store_backend

    @property
    def embedding_model_path(self) -> Path:
        return Path(self.embedding_model)

    @property
    def reranker_model_path(self) -> Path:
        return Path(self.reranker_model)

    @property
    def embedding_dimension(self) -> int:
        return self.embedding_dim

    @property
    def opensearch_username(self) -> str:
        return self.opensearch_user

    @property
    def enable_reranker(self) -> bool:
        return self.reranker_enabled

    @property
    def rerank_top_k(self) -> int:
        return self.reranker_top_k


    def validate(self, require_reranker: bool = False) -> None:
        if self.document_store_backend not in {"opensearch","in_memory"}:
            raise ValueError(
                "DOCUMENT_STORE_BACKEND must be either "
                "'opensearch' or 'in_memory'."
            )
        if not self.embedding_model_path.is_dir():
            raise FileNotFoundError(
                "Embedding model directory missing: "
                f"{self.embedding_model_path}"
            )

        if require_reranker and not self.reranker_model_path.is_dir():
            raise FileNotFoundError(
                "Reranker model was not found at "
                f"{self.reranker_model_path}"
            )

        if self.embedding_dimension <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be positive.")

        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive.")

        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP cannot be negative.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
            )

settings = Settings()

def get_settings() -> Settings:
    return settings
