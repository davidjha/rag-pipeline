from __future__ import annotations

from typing import Any

from haystack import Document, Pipeline
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack_integrations.components.retrievers.opensearch import (
    OpenSearchBM25Retriever,
    OpenSearchEmbeddingRetriever,
)

from config import Settings, get_settings
from rag.document_store import create_document_store
from rag.embedders import (
    create_query_embedder,
    create_reranker,
)


def build_retrieval_pipeline(
    settings: Settings | None = None,
    *,
    document_store=None
) -> Pipeline:
    settings = settings or get_settings()
    settings.validate(
        require_reranker=settings.enable_reranker
    )

    if document_store is None:
        document_store = create_document_store(settings)
    query_embedder = create_query_embedder(settings)

    if settings.document_store_backend == "in_memory":
        bm25_retriever = InMemoryBM25Retriever(
            document_store=document_store,
            top_k=settings.bm25_top_k,
        )

        embedding_retriever = InMemoryEmbeddingRetriever(
            document_store=document_store,
            top_k=settings.embedding_top_k,
        )

    elif settings.document_store_backend == "opensearch":
        from haystack_integrations.components.retrievers.opensearch import(
            OpenSearchBM25Retriever,
            OpenSearchEmbeddingRetriever
        )

        bm25_retriever = OpenSearchBM25Retriever(
            document_store=document_store,
            top_k=settings.bm25_top_k,
        )

        embedding_retriever = OpenSearchEmbeddingRetriever(
            document_store=document_store,
            top_k=settings.embedding_top_k,
        )

    else:
        raise ValueError(
            "Unsupported document-store backend: "
            f"{settings.document_store_backend}"
        )

    joiner = DocumentJoiner(
        join_mode="reciprocal_rank_fusion",
    )

    pipeline = Pipeline()

    pipeline.add_component(
        "query_embedder",
        query_embedder,
    )
    pipeline.add_component(
        "bm25_retriever",
        bm25_retriever,
    )
    pipeline.add_component(
        "embedding_retriever",
        embedding_retriever,
    )
    pipeline.add_component(
        "joiner",
        joiner,
    )

    pipeline.connect(
        "query_embedder.embedding",
        "embedding_retriever.query_embedding",
    )
    pipeline.connect(
        "bm25_retriever.documents",
        "joiner.documents",
    )
    pipeline.connect(
        "embedding_retriever.documents",
        "joiner.documents",
    )

    if settings.enable_reranker:
        pipeline.add_component(
            "reranker",
            create_reranker(settings),
        )

        pipeline.connect(
            "joiner.documents",
            "reranker.documents",
        )

    return pipeline

def run_retrieval_pipeline(
    pipeline: Pipeline,
    query: str,
    settings: Settings,
    top_k: int,
    *,
    candidate_k: int | None = None,
    filters: dict[str, Any] | None = None,
) -> list[Document]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if candidate_k is not None and candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k.")

    data: dict[str, Any] = {
        "query_embedder": {
            "text": query,
        },
        "bm25_retriever": {
            "query": query,
            "filters": filters,
        },
        "embedding_retriever": {
            "filters": filters,
        },
    }

    if candidate_k is not None:
        data["bm25_retriever"]["top_k"] = candidate_k
        data["embedding_retriever"]["top_k"] = candidate_k
        data["joiner"] = {"top_k": candidate_k}

    if settings.enable_reranker:
        data["reranker"] = {
            "query": query,
            "top_k": top_k,
        }

    result = pipeline.run(data=data)

    if settings.enable_reranker:
        return result["reranker"]["documents"]

    return result["joiner"]["documents"][:top_k]

def retrieve(
    query: str,
    settings: Settings | None = None,
    top_k: int | None = None,
) -> list[Document]:
    settings = settings or get_settings()
    pipeline = build_retrieval_pipeline(settings)

    final_top_k = top_k or settings.rerank_top_k

    return run_retrieval_pipeline(
        pipeline=pipeline,
        query=query,
        settings=settings,
        top_k=final_top_k,
    )
