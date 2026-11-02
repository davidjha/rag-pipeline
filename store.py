from config import Settings


def create_store(settings: Settings):
    backend = settings.store_backend.lower()

    if backend == "in_memory":
        from haystack.document_stores.in_memory import InMemoryDocumentStore

        return InMemoryDocumentStore(
            embedding_similarity_function="cosine",
        )

    if backend == "opensearch":
        from haystack_integrations.document_stores.opensearch import (
            OpenSearchDocumentStore,
        )

        kwargs = {}
        if settings.opensearch_ca_certs:
            kwargs["ca_certs"] = settings.opensearch_ca_certs

        auth = None
        if settings.opensearch_user:
            auth = (
                settings.opensearch_user,
                settings.opensearch_password,
            )

        return OpenSearchDocumentStore(
            hosts=settings.opensearch_url,
            index=settings.opensearch_index,
            embedding_dim=settings.embedding_dim,
            http_auth=auth,
            use_ssl=settings.opensearch_url.startswith("https://"),
            verify_certs=settings.opensearch_verify_certs,
            method={
                "name": "hnsw",
                "engine": "lucene",
                "space_type": "cosinesimil",
            },
            **kwargs,
        )

    raise ValueError(f"Unsupported STORE_BACKEND: {backend}")


def create_retrievers(store, backend: str):
    if backend.lower() == "in_memory":
        from haystack.components.retrievers.in_memory import (
            InMemoryBM25Retriever,
            InMemoryEmbeddingRetriever,
        )

        return (
            InMemoryBM25Retriever(document_store=store),
            InMemoryEmbeddingRetriever(document_store=store),
        )

    from haystack_integrations.components.retrievers.opensearch import (
        OpenSearchBM25Retriever,
        OpenSearchEmbeddingRetriever,
    )

    return (
        OpenSearchBM25Retriever(document_store=store),
        OpenSearchEmbeddingRetriever(document_store=store),
    )
