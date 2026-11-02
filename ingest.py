from __future__ import annotations

# Load offline configuration before importing Haystack components.
from config import get_settings

import argparse
import json

from rag.files import collect_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively discover and index local documents."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching files without loading models or writing data.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files/directories; defaults to DOCUMENTS_ROOT.",
    )
    args = parser.parse_args()

    settings = get_settings()
    paths = args.paths or [settings.documents_root]
    files = collect_files(paths)

    print(f"Found {len(files)} supported file(s):")
    for path in files:
        print(f"  {path}")

    if args.dry_run:
        print("Dry run complete; no models loaded or documents written.")
        return

    if not files:
        raise SystemExit("No supported documents were found.")

    # Defer these imports so dry-run remains lightweight.
    from rag.document_store import create_document_store
    from rag.services import Ingestor

    settings.validate()
    store = create_document_store(settings)
    ingestor = Ingestor(store, settings)

    result = ingestor.run_files(files)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
