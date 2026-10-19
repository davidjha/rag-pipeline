from __future__ import annotations

from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack_integrations.components.rankers.sentence_transformers import (
    SentenceTransformersSimilarityRanker,
)

from rag.config import Settings, get_settings


def create_document_embedder(
    settings: Settings | None = None,
) -> SentenceTransformersDocumentEmbedder:
    settings = settings or get_settings()
    settings.validate()

    return SentenceTransformersDocumentEmbedder(
        model=str(settings.embedding_model_path),
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=settings.normalize_embeddings,
    )


def create_query_embedder(
    settings: Settings | None = None,
) -> SentenceTransformersTextEmbedder:
    settings = settings or get_settings()
    settings.validate()

    return SentenceTransformersTextEmbedder(
        model=str(settings.embedding_model_path),
        normalize_embeddings=settings.normalize_embeddings,
    )


def create_reranker(
    settings: Settings | None = None,
) -> SentenceTransformersSimilarityRanker:
    settings = settings or get_settings()
    settings.validate(require_reranker=True)

    return SentenceTransformersSimilarityRanker(
        model=str(settings.reranker_model_path),
        top_k=settings.rerank_top_k,
    )
