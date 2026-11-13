"""Offline tests: TF-IDF embeddings + no LLM, so they run anywhere without keys or downloads."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.chunking import chunk_documents, split_sections  # noqa: E402
from app.config import BASE_DIR, Settings  # noqa: E402
from app.loaders import Document  # noqa: E402
from app.main import create_app  # noqa: E402
from app.service import RAGService  # noqa: E402


@pytest.fixture()
def settings(tmp_path):
    return Settings(
        sample_dir=BASE_DIR / "data" / "sample_docs",
        upload_dir=tmp_path / "uploads",
        index_dir=tmp_path / "index",
        embedding_backend="tfidf",
        llm_provider="none",
    )


@pytest.fixture()
def service(settings):
    svc = RAGService(settings)
    svc.build_index()
    return svc


# ---------------------------------------------------------------- chunking
def test_headings_inside_code_fences_are_ignored():
    text = "# Title\n\n## Real heading\nSome body text here.\n\n```python\n# not a heading\nx = 1\n```\n"
    sections = split_sections(text)
    assert len(sections) == 1
    assert sections[0][0] == ["Title", "Real heading"]
    assert "# not a heading" in sections[0][1]


def test_question_chunks_carry_metadata():
    doc = Document("t.md", "demo", "## Section\n\n### Q: [hard] What is X?\n" + "Answer words. " * 20)
    chunks = chunk_documents([doc])
    assert len(chunks) == 1
    c = chunks[0]
    assert c.is_question and c.question == "What is X?" and c.difficulty == "hard"
    assert "[hard]" not in c.heading


def test_long_sections_are_split_with_overlap():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(10))
    chunks = chunk_documents([Document("big.md", "demo", "# Big\n\n" + body)], chunk_words=150, overlap=70)
    assert len(chunks) > 3
    assert all(len(c.text.split()) <= 150 + 70 for c in chunks)


# ---------------------------------------------------------------- retrieval
def test_hybrid_retrieval_finds_the_gil(service):
    hits = service.retriever.retrieve("global interpreter lock threads", k=3)
    assert hits and hits[0].chunk.topic == "python"
    assert "GIL" in hits[0].chunk.text


def test_topic_filter_restricts_results(service):
    hits = service.retriever.retrieve("explain indexes and joins", k=5, topic="sql-databases")
    assert hits and all(h.chunk.topic == "sql-databases" for h in hits)


def test_dense_and_bm25_modes_both_work(service):
    for mode in ("dense", "bm25", "hybrid"):
        assert service.retriever.retrieve("what is overfitting", k=3, mode=mode)


def test_ask_without_llm_is_extractive_with_sources(service):
    out = service.ask("What is Retrieval-Augmented Generation?")
    assert out["mode"] == "extractive" and out["sources"]
    assert "RAG" in out["answer"] or "retriev" in out["answer"].lower()


# ---------------------------------------------------------------- interview engine
def test_interview_flow_and_grading(service):
    engine = service.engine
    session, first = engine.start("python", "mixed", 2)
    assert first["index"] == 1 and first["question"]

    # A good answer (the reference itself) should outscore a vague one.
    good = engine.submit_answer(session.id, session.turns[0].reference)
    assert good["feedback"]["score"] >= 7
    assert not good["finished"] and good["next_question"]

    bad = engine.submit_answer(session.id, "I don't know")
    assert bad["feedback"]["score"] < 3
    assert bad["finished"] and bad["summary"]["average_score"] > 0
    assert len({t.question for t in session.turns}) == 2  # no repeats


def test_unknown_topic_and_finished_session_raise(service):
    with pytest.raises(ValueError):
        service.normalise_topic("does-not-exist")
    session, _ = service.engine.start("behavioral", "easy", 1)
    service.engine.submit_answer(session.id, "Situation, task, action, result.")
    with pytest.raises(ValueError):
        service.engine.submit_answer(session.id, "again")


# ---------------------------------------------------------------- HTTP API
def test_api_end_to_end(settings):
    app = create_app(settings)
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["status"] == "ok" and health["chunks"] > 20 and health["llm"] == "none"

        topics = {t["topic"] for t in client.get("/api/topics").json()["topics"]}
        assert {"python", "machine-learning", "genai-llm-rag"} <= topics

        ask = client.post("/api/ask", json={"question": "how does binary search work?"})
        assert ask.status_code == 200 and ask.json()["sources"]

        start = client.post("/api/interview/start", json={"topic": "python", "difficulty": "easy", "num_questions": 1})
        assert start.status_code == 200
        sid = start.json()["session_id"]
        done = client.post("/api/interview/answer", json={"session_id": sid, "answer": "A list is mutable, a tuple is immutable and hashable."})
        assert done.status_code == 200 and done.json()["finished"] is True
        assert client.get(f"/api/interview/{sid}/summary").status_code == 200

        assert client.post("/api/interview/answer", json={"session_id": "nope", "answer": "x"}).status_code == 404
        assert client.post("/api/ask", json={"question": "hi", "topic": "nonsense-topic"}).status_code in (400, 422)
        assert client.get("/").status_code == 200  # frontend served


def test_upload_adds_new_topic(settings):
    app = create_app(settings)
    with TestClient(app) as client:
        notes = (
            b"# Rust\n\n### Q: [medium] What is ownership in Rust?\n"
            b"Ownership means each value has a single owner; when the owner goes out of scope the value is dropped, "
            b"and borrowing lets other code use the value through references without taking ownership.\n"
        )
        r = client.post("/api/upload", files={"file": ("rust.md", notes, "text/markdown")}, data={"topic": "rust"})
        assert r.status_code == 200, r.text
        assert "rust" in {t["topic"] for t in client.get("/api/topics").json()["topics"]}

        hit = client.post("/api/ask", json={"question": "what is ownership and borrowing", "topic": "rust"}).json()
        assert hit["sources"] and hit["sources"][0]["topic"] == "rust"

        bad = client.post("/api/upload", files={"file": ("virus.exe", b"x", "application/octet-stream")})
        assert bad.status_code == 400
