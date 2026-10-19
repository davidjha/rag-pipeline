from __future__ import annotations

from haystack import Pipeline
from haystack_integrations.components.retrievers.opensearch import (
    OpenSearchHybridRetriever,
)

from rag.config import Settings, get_settings
from rag.document_store import create_document_store
from rag.embedders import create_query_embedder, create_reranker


def build_retrieval_pipeline(
    settings: Settings | None = None,
) -> Pipeline:
    settings = settings or get_settings()
    settings.validate(
        require_reranker=settings.enable_reranker
    )

    document_store = create_document_store(settings)
    query_embedder = create_query_embedder(settings)

    hybrid_retriever = OpenSearchHybridRetriever(
        document_store=document_store,
        embedder=query_embedder,
        top_k_bm25=settings.bm25_top_k,
        top_k_embedding=settings.embedding_top_k,
        join_mode="reciprocal_rank_fusion",
    )

    pipeline = Pipeline()
    pipeline.add_component(
        "hybrid_retriever",
        hybrid_retriever,
    )

    if settings.enable_reranker:
        pipeline.add_component(
            "reranker",
            create_reranker(settings),
        )

        pipeline.connect(
            "hybrid_retriever.documents",
            "reranker.documents",
        )

    return pipeline


def retrieve(
    query: str,
    settings: Settings | None = None,
    top_k: int | None = None,
) -> list:
    settings = settings or get_settings()
    pipeline = build_retrieval_pipeline(settings)

    final_top_k = top_k or settings.rerank_top_k

    if settings.enable_reranker:
        result = pipeline.run(
            data={
                "hybrid_retriever": {
                    "query": query,
                },
                "reranker": {
                    "query": query,
                    "top_k": final_top_k,
                },
            }
        )

        return result["reranker"]["documents"]

    result = pipeline.run(
        data={
            "hybrid_retriever": {
                "query": query,
            }
        }
    )

    return result["hybrid_retriever"]["documents"][
        :final_top_k
    ]
