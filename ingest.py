from __future__ import annotations

import argparse
from pathlib import Path
import sys

from rag.config import get_settings
from rag.document_store import create_document_store
from rag.indexing_pipeline import build_indexing_pipeline


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".docx",
    ".pptx",
}

def is_hidden_or_temporary(path: Path) -> bool:
    return (
        any(part.startswith(".") for part in path.parts)
        or path.name.startswith("~$")
        or path.name.endswith(".tmp")
    )


def collect_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []

    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Input path does not exist: {path}"
            )

        if path.is_file():
            if (
                path.suffix.lower() in SUPPORTED_EXTENSIONS
                and not is_hidden_or_temporary(path)
            ):
                files.append(path)
            else:
                print(
                    f"Skipping unsupported or temporary file: {path}",
                    file=sys.stderr,
                )
            continue

        for candidate in path.rglob("*"):
            if (
                candidate.is_file()
                and candidate.suffix.lower()
                in SUPPORTED_EXTENSIONS
                and not is_hidden_or_temporary(candidate)
            ):
                files.append(candidate)

    return sorted(set(files))

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert, split, embed, and index local documents "
            "in OpenSearch."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching files without ingesting them.",
    )

    parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to ingest.",
    )

    args = parser.parse_args()
    files = collect_files(args.paths)

    if not files:
        raise SystemExit("No supported documents were found.")

    print(f"Found {len(files)} document file(s):")
    for file_path in files:
        print(f"  - {file_path}")

    if args.dry_run:
        print()
        print("Dry rune complete; no documents were ingest.")
        return

    settings = get_settings()
    settings.validate()

    pipeline = build_indexing_pipeline(settings)

    result = pipeline.run(
        data={
            "converter": {
                "sources": files,
            }
        }
    )

    documents_written = result["writer"][
        "documents_written"
    ]

    document_store = create_document_store(settings)

    print()
    print(f"Chunks written: {documents_written}")
    print(
        "Total chunks in index: "
        f"{document_store.count_documents()}"
    )


if __name__ == "__main__":
    main()
