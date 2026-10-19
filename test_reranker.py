from haystack import Document

from rag.config import get_settings
from rag.embedders import create_reranker

# Using the example 

def main() -> None:
    settings = get_settings()
    ranker = create_reranker(settings)
    ranker.warm_up()

    result = ranker.run(
        query="How quickly must a security incident be reported?",
        documents=[
            Document(
                content=(
                    "Security incidents must be reported within "
                    "one hour of discovery."
                )
            ),
            Document(
                content=(
                    "Employees may work remotely up to three days "
                    "per week."
                )
            ),
        ],
        top_k=2,
    )

    for document in result["documents"]:
        print(f"Score: {document.score}")
        print(f"Text: {document.content}")
        print()


if __name__ == "__main__":
    main()
