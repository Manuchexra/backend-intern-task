from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, StrictInt, field_validator

from app import llm  # noqa: F401  (you will need it)
from app.chunking import chunk_text  # noqa: F401  (you will need it)


class DocumentInput(BaseModel):
    id: str
    text: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("id must not be empty")
        return value

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        return value


class IngestInput(BaseModel):
    documents: list[DocumentInput] = Field(min_length=1)


class AskInput(BaseModel):
    question: str
    top_k: StrictInt = Field(default=3, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value


def create_app() -> FastAPI:
    """Return a NEW app with an EMPTY in-memory store.

    The tests call create_app() once per test, so state must not leak
    between app instances (no module-level store).
    """
    app = FastAPI()
    store: dict[str, list[dict]] = {}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ingest")
    def ingest(payload: IngestInput):
        total_chunks = 0

        for document in payload.documents:
            chunks = chunk_text(document.text)
            embedded_chunks = [
                {
                    "chunk_id": f"{document.id}:{index}",
                    "text": chunk,
                    "vector": llm.embed(chunk),
                }
                for index, chunk in enumerate(chunks)
            ]
            store[document.id] = embedded_chunks
            total_chunks += len(embedded_chunks)

        return {"ingested": len(payload.documents), "chunks": total_chunks}

    @app.post("/ask")
    def ask(payload: AskInput):
        if not store:
            raise HTTPException(status_code=404, detail="No documents ingested")

        question_vector = llm.embed(payload.question)
        scored_chunks: list[tuple[float, str, dict]] = []

        for doc_id, chunks in store.items():
            for chunk in chunks:
                score = sum(
                    question_value * chunk_value
                    for question_value, chunk_value in zip(
                        question_vector, chunk["vector"]
                    )
                )
                scored_chunks.append((score, doc_id, chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        best_chunks = scored_chunks[: payload.top_k]
        context_texts = [chunk["text"] for _, _, chunk in best_chunks]

        sources = [
            {
                "doc_id": doc_id,
                "chunk_id": chunk["chunk_id"],
                "score": score,
                "text": chunk["text"],
            }
            for score, doc_id, chunk in best_chunks
        ]

        return {
            "answer": llm.generate(payload.question, context_texts),
            "sources": sources,
        }

    @app.get("/documents")
    def documents():
        return {
            "documents": [
                {"id": doc_id, "chunks": len(store[doc_id])}
                for doc_id in sorted(store)
            ]
        }

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str):
        if doc_id not in store:
            raise HTTPException(status_code=404, detail="Document not found")
        del store[doc_id]
        return Response(status_code=204)

    return app


app = create_app()
