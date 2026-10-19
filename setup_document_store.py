from __future__ import annotations

from rag.config import get_settings
from rag.document_store import create_document_store


def main() -> None:
    settings = get_settings()
    settings.validate(
        require_reranker=settings.enable_reranker
    )

    document_store = create_document_store(settings)

    print("OpenSearch document store is ready.")
    print(f"Index: {settings.opensearch_index}")
    print(
        "Embedding model: "
        f"{settings.embedding_model_path.resolve()}"
    )
    print(
        "Embedding dimension: "
        f"{settings.embedding_dimension}"
    )
    print(
        "Current document count: "
        f"{document_store.count_documents()}"
    )


if __name__ == "__main__":
    main()
