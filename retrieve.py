from __future__ import annotations

import argparse
from textwrap import shorten

from rag.config import get_settings
from rag.retrieval_pipeline import retrieve


def get_source(meta: dict) -> str:
    for key in (
        "file_path",
        "source_path",
        "filename",
        "name",
    ):
        value = meta.get(key)
        if value:
            return str(value)

    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run hybrid BM25/vector retrieval and optional "
            "MS MARCO reranking."
        )
    )

    parser.add_argument(
        "query",
        help="Question or search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of final chunks to return.",
    )

    args = parser.parse_args()
    settings = get_settings()

    documents = retrieve(
        query=args.query,
        settings=settings,
        top_k=args.top_k,
    )

    if not documents:
        print("No matching documents were found.")
        return

    print(f"Query: {args.query}")
    print(f"Results: {len(documents)}")
    print()

    for position, document in enumerate(documents, start=1):
        source = get_source(document.meta)
        page_number = document.meta.get("page_number")
        split_id = document.meta.get("split_id")

        print("=" * 80)
        print(f"Rank: {position}")
        print(f"Score: {document.score}")
        print(f"Document ID: {document.id}")
        print(f"Source: {source}")

        if page_number is not None:
            print(f"Page: {page_number}")

        if split_id is not None:
            print(f"Split: {split_id}")

        print()
        print(
            shorten(
                document.content or "",
                width=1_000,
                placeholder=" ...",
            )
        )
        print()


if __name__ == "__main__":
    main()
