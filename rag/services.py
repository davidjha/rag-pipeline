from dataclasses import replace
from pathlib import Path

from config import Settings
from rag.document_store import persist_document_store
from rag.embedders import create_reranker
from rag.files import collect_files
from rag.indexing_pipeline import build_indexing_pipeline
from rag.retrieval_pipeline import (
    build_retrieval_pipeline,
    run_retrieval_pipeline,
)


class Ingestor:
    def __init__(self, document_store, settings: Settings):
        self.store = document_store
        self.settings = settings
        self.root = settings.documents_root

        # Do not silently create an empty directory if configuration is wrong.
        self.pipeline = build_indexing_pipeline(
            settings,
            document_store=document_store,
        )

    def run(self, relative_path: str = ".", dry_run: bool = False):
        supplied_path = Path(relative_path)

        if supplied_path.is_absolute():
            raise ValueError(
                "API ingestion paths must be relative to DOCUMENTS_ROOT."
            )

        target = (self.root / supplied_path).resolve()

        if not target.is_relative_to(self.root):
            raise ValueError(
                "Path must remain inside DOCUMENTS_ROOT."
            )

        if not target.exists():
            raise ValueError(
                f"Ingestion path does not exist: {target}"
            )

        files = collect_files([target])

        # Reject symlinks that resolve outside the configured directory.
        if any(not path.is_relative_to(self.root) for path in files):
            raise ValueError(
                "An input file resolves outside DOCUMENTS_ROOT."
            )

        return self.run_files(files, dry_run=dry_run)

    def run_files(self, files: list[Path], dry_run: bool = False):
        summary = {
            "files_discovered": len(files),
            "chunks_written": 0,
            "dry_run": dry_run,
            "conversion_failed": [],
            "unclassified": [],
        }

        if dry_run:
            return {
                **summary,
                "files": [str(path) for path in files],
            }

        if not files:
            return {
                **summary,
                "message": "No supported files found.",
            }

        result = self.pipeline.run(
            data={
                "converter": {
                    "sources": files,
                },
            },
            include_outputs_from={"converter"},
        )

        summary["chunks_written"] = result["writer"]["documents_written"]

        converter_result = result.get("converter", {})

        summary["conversion_failed"] = [
            str(item)
            for item in converter_result.get("failed", [])
        ]
        summary["unclassified"] = [
            str(item)
            for item in converter_result.get("unclassified", [])
        ]

        persist_document_store(
            document_store=self.store,
            settings=self.settings,
        )

        summary["total_chunks"] = self.store.count_documents()

        if summary["chunks_written"] == 0:
            summary["message"] = (
                "Files were discovered, but no chunks were written. "
                "Inspect conversion results and extracted text."
            )

        return summary


class Reranker:
    def __init__(self, settings: Settings):
        self.component = None

        if settings.reranker_enabled:
            self.component = create_reranker(settings)
            self.component.warm_up()

    def run(self, query, documents, top_k):
        if not documents:
            return []

        if self.component is None:
            return documents[:top_k]

        return self.component.run(
            query=query,
            documents=documents,
            top_k=top_k,
        )["documents"]


class Retriever:
    def __init__(self, document_store, settings: Settings, reranker):
        self.reranker = reranker

        # The shared retrieval pipeline supplies hybrid candidates.
        # Reranking is invoked separately to support per-request toggling
        # and the standalone /rerank endpoint without loading another ranker.
        self.retrieval_settings = replace(
            settings,
            reranker_enabled=False,
        )

        self.pipeline = build_retrieval_pipeline(
            self.retrieval_settings,
            document_store=document_store,
        )
        self.pipeline.warm_up()

    def run(
        self,
        query,
        top_k=5,
        candidate_k=20,
        filters=None,
        rerank=True,
    ):
        if candidate_k < top_k:
            raise ValueError("candidate_k must be at least top_k.")

        candidates = run_retrieval_pipeline(
            pipeline=self.pipeline,
            query=query,
            settings=self.retrieval_settings,
            top_k=candidate_k,
            candidate_k=candidate_k,
            filters=filters,
        )

        if rerank:
            return self.reranker.run(
                query=query,
                documents=candidates,
                top_k=top_k,
            )

        return candidates[:top_k]
