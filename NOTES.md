I used an in-memory dictionary to isolate document chunks and embeddings per app instance.
Chunking uses whitespace splitting and overlapping word windows, while retrieval ranks chunks by dot product.
I used FastAPI/Pydantic validation and preserved the standard 422, 404, and 204 response behavior.
With more time, I would add focused tests for replacement, deletion, validation, and edge-case chunking.
AI tools used: Copilot coding agent.
