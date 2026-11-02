from __future__ import annotations

from config import get_settings

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test hybrid retrieval against the configured store."
    )
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    if args.top_k <= 0 or args.candidate_k < args.top_k:
        parser.error("Require candidate-k >= top-k > 0.")

    from dataclasses import replace

    from rag.document_store import create_document_store
    from rag.services import Reranker, Retriever

    settings = get_settings()

    if args.no_rerank:
        settings = replace(settings, reranker_enabled=False)

    settings.validate(
        require_reranker=settings.reranker_enabled
    )

    store = create_document_store(settings)
    ranker = Reranker(settings)
    retriever = Retriever(store, settings, ranker)

    documents = retriever.run(
        query=args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rerank=not args.no_rerank,
    )

    print(json.dumps(
        [
            {
                "id": document.id,
                "content": document.content,
                "metadata": document.meta,
                "score": document.score,
            }
            for document in documents
        ],
        indent=2,
        default=str,
    ))


if __name__ == "__main__":
    main()
