"""A few public tests to get you started.

The grader runs a larger hidden test suite against the same spec in README.md.
Passing these tests does NOT mean you will pass the hidden ones.
"""

import pytest
from fastapi.testclient import TestClient

from app.chunking import chunk_text
from app.main import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_chunk_text_basic():
    text = " ".join(f"w{i}" for i in range(12))
    assert chunk_text(text, size=5, overlap=2) == [
        "w0 w1 w2 w3 w4",
        "w3 w4 w5 w6 w7",
        "w6 w7 w8 w9 w10",
        "w9 w10 w11",
    ]


def test_ingest_then_ask(client):
    r = client.post(
        "/ingest",
        json={"documents": [{"id": "cats", "text": "Cats are small furry pets that purr."}]},
    )
    assert r.status_code == 200
    assert r.json() == {"ingested": 1, "chunks": 1}

    r = client.post("/ask", json={"question": "Which pets purr?"})
    assert r.status_code == 200
    body = r.json()
    assert body["sources"][0]["doc_id"] == "cats"
    assert body["sources"][0]["chunk_id"] == "cats:0"


def test_replacement_documents_and_delete(client):
    first = client.post(
        "/ingest",
        json={"documents": [{"id": "b", "text": "One two three four"}]},
    )
    assert first.status_code == 200

    client.post(
        "/ingest",
        json={
            "documents": [
                {"id": "a", "text": "Alpha beta"},
                {"id": "b", "text": "Replacement"},
            ]
        },
    )

    documents = client.get("/documents")
    assert documents.status_code == 200
    assert documents.json() == {
        "documents": [
            {"id": "a", "chunks": 1},
            {"id": "b", "chunks": 1},
        ]
    }

    deleted = client.delete("/documents/b")
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert client.get("/documents").json() == {
        "documents": [{"id": "a", "chunks": 1}]
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"documents": []},
        {"documents": [{"id": "", "text": "text"}]},
        {"documents": [{"id": "doc", "text": "   "}]},
    ],
)
def test_ingest_validation(client, payload):
    assert client.post("/ingest", json=payload).status_code == 422


@pytest.mark.parametrize("top_k", [0, 11, "3", True])
def test_ask_top_k_validation(client, top_k):
    assert client.post("/ask", json={"question": "question", "top_k": top_k}).status_code == 422
