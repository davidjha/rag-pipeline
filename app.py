# Import config first so offline flags are set before Haystack imports.
from config import settings

import logging
import secrets
from contextlib import asynccontextmanager
from threading import RLock
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from haystack import Document
from openai import APIError
from pydantic import BaseModel, Field, model_validator

from generate import Generator
from rag.document_store import create_document_store
from rag.services import Ingestor, Reranker, Retriever

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.rag_auth_enabled and not settings.api_key:
        raise RuntimeError("Set RAG_API_KEY when RAG_AUTH_ENABLED=true before starting the service")

    settings.validate(
        require_reranker=settings.reranker_enabled
    )

    store = create_document_store(settings)

    # Initialize/validate the configured document store at startup.
    store.count_documents()

    ranker = Reranker(settings)

    app.state.store = store
    app.state.lock = RLock()
    app.state.ingestor = Ingestor(store, settings)
    app.state.ranker = ranker
    app.state.retriever = Retriever(store, settings, ranker)
    app.state.generator = Generator(settings)

    try:
        yield
    finally:
        app.state.generator.close()


def authorize(
    x_api_key: str = Header(default="", alias="X-API-Key"),
):
    if not settings.rag_auth_enabled:
        return

    if not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")


app = FastAPI(
    title="Local Haystack RAG API",
    lifespan=lifespan,
    dependencies=[Depends(authorize)],
)


class IngestRequest(BaseModel):
    path: str = "."
    dry_run: bool = False


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)
    candidate_k: int = Field(default=20, ge=1, le=100)
    filters: dict[str, Any] | None = None
    rerank: bool = True

    @model_validator(mode="after")
    def validate_candidate_count(self):
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be at least top_k")
        return self


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(RetrievalRequest):
    history: list[HistoryMessage] = Field(
        default_factory=list,
        max_length=12,
    )
    # Useful for follow-up questions that need a standalone search query.
    retrieval_query: str | None = Field(
        default=None,
        min_length=1,
        max_length=4000,
    )
    max_tokens: int = Field(default=512, ge=1, le=2048)


class RerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    texts: list[str] = Field(min_length=1, max_length=100)
    top_k: int = Field(default=5, ge=1, le=100)


def serialize_documents(documents):
    return [
        {
            "reference": number,
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.meta,
            "score": doc.score,
        }
        for number, doc in enumerate(documents, start=1)
    ]


def retrieve_documents(state, body):
    query = getattr(body, "retrieval_query", None) or body.query

    return state.retriever.run(
        query=query,
        top_k=body.top_k,
        candidate_k=body.candidate_k,
        filters=body.filters,
        rerank=body.rerank,
    )


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "store_backend": settings.store_backend,
        "reranker_enabled": settings.reranker_enabled,
    }


@app.post("/ingest")
def ingest(body: IngestRequest, request: Request):
    try:
        with request.app.state.lock:
            return request.app.state.ingestor.run(
                relative_path=body.path,
                dry_run=body.dry_run
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/retrieve")
def retrieve(body: RetrievalRequest, request: Request):
    with request.app.state.lock:
        documents = retrieve_documents(request.app.state, body)

    return {"documents": serialize_documents(documents)}


@app.post("/rerank")
def rerank(body: RerankRequest, request: Request):
    documents = [
        Document(id=str(number), content=text)
        for number, text in enumerate(body.texts)
    ]

    with request.app.state.lock:
        documents = request.app.state.ranker.run(
            query=body.query,
            documents=documents,
            top_k=body.top_k,
        )

    return {"documents": serialize_documents(documents)}

@app.post("/store/stats")
def store_stats(request: Request):
    with request.app.state.lock:
        count = request.app.state.store.count_documents()
    return {
        "backend": settings.store_backend,
        "document_count": count,
        "documents_root": str(settings.documents_root),
        "snapshot_enabled": (
            settings.store_backend == "in_memory"
            and settings.in_memory_persist
        ),
    }

@app.post("/chat")
def chat(body: ChatRequest, request: Request):
    state = request.app.state

    with state.lock:
        documents = retrieve_documents(state, body)

    # Generation does not hold the retrieval/ingestion lock.
    try:
        result = state.generator.run(
            question=body.query,
            documents=documents,
            history=[message.model_dump() for message in body.history],
            max_tokens=body.max_tokens,
        )
    except APIError as exc:
        logger.exception("Inference request failed")
        raise HTTPException(
            status_code=502,
            detail="Inference server request failed",
        ) from exc

    return {
        **result,
        "sources": serialize_documents(documents),
    }
