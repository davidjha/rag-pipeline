from __future__ import annotations

from haystack_integrations.document_stores.opensearch import (
    OpenSearchDocumentStore,
)

from rag.config import Settings, get_settings


def create_document_store(
    settings: Settings | None = None,
) -> OpenSearchDocumentStore:
    settings = settings or get_settings()

    if not settings.opensearch_password:
        raise ValueError(
            "OPENSEARCH_PASSWORD is empty. Set it in .env or the environment."
        )

    # HNSW provides approximate nearest-neighbor vector search.
    # Cosine similarity is appropriate for normalized MPNet embeddings.
    hnsw_method = {
        "name": "hnsw",
        "engine": "lucene",
        "space_type": "cosinesimil",
        "parameters": {
            "ef_construction": 128,
            "m": 16,
        },
    }

    document_store = OpenSearchDocumentStore(
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

    return document_store
