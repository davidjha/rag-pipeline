from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


@dataclass(frozen=True)
class Settings:
    document_store_backend: str
    
    in_memory_store_path: Path
    in_memory_persist: bool

    opensearch_url: str
    opensearch_username: str
    opensearch_password: str
    opensearch_verify_certs: bool
    opensearch_index: str
    opensearch_timeout: int

    embedding_model_path: Path
    embedding_dimension: int
    embedding_batch_size: int
    normalize_embeddings: bool

    reranker_model_path: Path
    enable_reranker: bool

    chunk_size: int
    chunk_overlap: int

    bm25_top_k: int
    embedding_top_k: int
    rerank_top_k: int

    def validate(self, require_reranker: bool = False) -> None:
        if self.document_store_backend not in {
            "opensearch",
            "in_memory",
        }:
            raise ValueError(
                "DOCUMENT_STORE_BACKEND must be either "
                "'opensearch' or 'in_memory'."
            )
        if not self.embedding_model_path.exists():
            raise FileNotFoundError(
                "Embedding model was not found at "
                f"{self.embedding_model_path.resolve()}"
            )

        if require_reranker and not self.reranker_model_path.exists():
            raise FileNotFoundError(
                "Reranker model was not found at "
                f"{self.reranker_model_path.resolve()}"
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()

    settings = Settings(
        document_store_backend=os.getenv(
            "DOCUMENT_STORE_BACKEND",
            "opensearch",
        ).strip().lower(),
        in_memory_store_path=Path(
            os.getenv(
                "IN_MEMORY_STORE_PATH",
                "./data/in-memory-document-store.json",
            )
        ),
        in_memory_persist=_as_bool(
            os.getenv("IN_MEMORY_PERSIST"),
            default=True,
        ),
        opensearch_url=os.getenv(
            "OPENSEARCH_URL",
            "https://localhost:9200",
        ),
        opensearch_username=os.getenv(
            "OPENSEARCH_USERNAME",
            "admin",
        ),
        opensearch_password=os.getenv(
            "OPENSEARCH_PASSWORD",
            "",
        ),
        opensearch_verify_certs=_as_bool(
            os.getenv("OPENSEARCH_VERIFY_CERTS"),
            default=False,
        ),
        opensearch_index=os.getenv(
            "OPENSEARCH_INDEX",
            "rag-documents",
        ),
        opensearch_timeout=int(
            os.getenv("OPENSEARCH_TIMEOUT", "60")
        ),
        embedding_model_path=Path(
            os.getenv(
                "EMBEDDING_MODEL_PATH",
                "./models/all-mpnet-base-v2",
            )
        ),
        embedding_dimension=int(
            os.getenv("EMBEDDING_DIMENSION", "768")
        ),
        embedding_batch_size=int(
            os.getenv("EMBEDDING_BATCH_SIZE", "16")
        ),
        normalize_embeddings=_as_bool(
            os.getenv("NORMALIZE_EMBEDDINGS"),
            default=True,
        ),
        reranker_model_path=Path(
            os.getenv(
                "RERANKER_MODEL_PATH",
                "./models/ms-marco-MiniLM-L-6-v2",
            )
        ),
        enable_reranker=_as_bool(
            os.getenv("ENABLE_RERANKER"),
            default=True,
        ),
        chunk_size=int(os.getenv("CHUNK_SIZE", "250")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "40")),
        bm25_top_k=int(os.getenv("BM25_TOP_K", "20")),
        embedding_top_k=int(
            os.getenv("EMBEDDING_TOP_K", "20")
        ),
        rerank_top_k=int(os.getenv("RERANK_TOP_K", "5")),
    )

    return settings
