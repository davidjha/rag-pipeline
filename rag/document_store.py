from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


from haystack.document_stores.in_memory import (
    InMemoryDocumentStore,
)
from haystack_integrations.document_stores.opensearch import (
    OpenSearchDocumentStore,
)

from rag.config import Settings, get_settings


DocumentStore = InMemoryDocumentStore | OpenSearchDocumentStore


def _create_in_memory_store(
    settings: Settings,
) -> InMemoryDocumentStore:
    snapshot_path = settings.in_memory_store_path.expanduser().resolve()

    if (
        settings.in_memory_persist
        and snapshot_path.exists()
    ):
        print(
            "Loading in-memory document store snapshot from "
            f"{snapshot_path}"
        )

        return InMemoryDocumentStore.load_from_disk(
            str(snapshot_path)
        )

    return InMemoryDocumentStore(
        embedding_similarity_function="cosine",
    )


def _create_opensearch_store(
    settings: Settings,
) -> OpenSearchDocumentStore:
    if not settings.opensearch_password:
        raise ValueError(
            "OPENSEARCH_PASSWORD is required when "
            "DOCUMENT_STORE_BACKEND=opensearch."
        )

    hnsw_method = {
        "name": "hnsw",
        "engine": "lucene",
        "space_type": "cosinesimil",
        "parameters": {
            "ef_construction": 128,
            "m": 16,
        },
    }

    return OpenSearchDocumentStore(
        hosts=settings.opensearch_url,
        index=settings.opensearch_index,
        embedding_dim=settings.embedding_dimension,
        method=hnsw_method,
        create_index=True,
        http_auth=(
            settings.opensearch_username,
            settings.opensearch_password,
        ),
        use_ssl=settings.opensearch_url.startswith("https://"),
        verify_certs=settings.opensearch_verify_certs,
        timeout=settings.opensearch_timeout,
        return_embedding=False,
    )


@lru_cache(maxsize=4)
def create_document_store(
    settings: Settings | None = None,
) -> DocumentStore:
    """
    Create and cache one document-store instance per Settings value.

    Caching is especially important for the in-memory backend. If each
    pipeline received a new instance, ingested documents would not be
    visible to retrieval in the same process.
    """
    settings = settings or get_settings()
    settings.validate(
        require_reranker=False,
    )

    if settings.document_store_backend == "in_memory":
        return _create_in_memory_store(settings)

    if settings.document_store_backend == "opensearch":
        return _create_opensearch_store(settings)

    raise ValueError(
        "Unsupported document-store backend: "
        f"{settings.document_store_backend}"
    )


def persist_document_store(
    document_store: DocumentStore,
    settings: Settings | None = None,
) -> None:
    """
    Save an in-memory store so another Python process can load it.

    OpenSearch persists data itself, so this is a no-op for OpenSearch.
    """
    settings = settings or get_settings()

    if settings.document_store_backend != "in_memory":
        return

    if not settings.in_memory_persist:
        return

    if not isinstance(document_store, InMemoryDocumentStore):
        raise TypeError(
            "Expected InMemoryDocumentStore for the "
            "in_memory backend."
        )

    snapshot_path = settings.in_memory_store_path.expanduser().resolve()
    snapshot_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    document_store.save_to_disk(
        str(snapshot_path)
    )

    print(
        "Saved in-memory document store snapshot to "
        f"{snapshot_path}"
    )


def reset_cached_document_stores() -> None:
    """
    Primarily useful in tests after changing configuration.
    """
    create_document_store.cache_clear()
