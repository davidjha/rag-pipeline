from __future__ import annotations

from haystack import Pipeline
from haystack.components.converters import MultiFileConverter
from haystack.components.preprocessors import (
    DocumentCleaner,
    DocumentSplitter,
)
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

from rag.config import Settings, get_settings
from rag.document_store import create_document_store
from rag.embedders import create_document_embedder


def build_indexing_pipeline(
    settings: Settings | None = None,
) -> Pipeline:
    settings = settings or get_settings()
    settings.validate()

    document_store = create_document_store(settings)
    document_embedder = create_document_embedder(settings)

    pipeline = Pipeline()

    pipeline.add_component(
        "converter",
        MultiFileConverter(),
    )

    pipeline.add_component(
        "cleaner",
        DocumentCleaner(),
    )

    pipeline.add_component(
        "splitter",
        DocumentSplitter(
            split_by="word",
            split_length=settings.chunk_size,
            split_overlap=settings.chunk_overlap,
        ),
    )

    pipeline.add_component(
        "document_embedder",
        document_embedder,
    )

    pipeline.add_component(
        "writer",
        DocumentWriter(
            document_store=document_store,
            policy=DuplicatePolicy.OVERWRITE,
        ),
    )

    pipeline.connect(
        "converter.documents",
        "cleaner.documents",
    )
    pipeline.connect(
        "cleaner.documents",
        "splitter.documents",
    )
    pipeline.connect(
        "splitter.documents",
        "document_embedder.documents",
    )
    pipeline.connect(
        "document_embedder.documents",
        "writer.documents",
    )

    return pipeline
