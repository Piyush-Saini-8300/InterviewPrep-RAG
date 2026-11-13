# 🎯 RAG Interview Prep

A Retrieval-Augmented Generation app that turns your own study notes into an interview coach.

- **Ask your notes** – questions answered from your documents, with cited sources.
- **Mock interviews** – questions drawn from your material, answers graded 0–10 with feedback, a model answer and a final summary.
- **Bring your own docs** – upload `.md`, `.txt`, `.pdf` or `.docx` and it re-indexes instantly.
- **Works with no API key** – offline mode uses TF-IDF + BM25 retrieval, extractive answers and a keyword/semantic grader. Add a key and the same app uses an LLM for answers, question generation and grading.

Ships with seven ready-made topics: Python, Data Structures & Algorithms, Machine Learning, GenAI/LLM/RAG, System Design, SQL & Databases and Behavioral (49 interview questions in total).

---

## Quick start

```bash
cd rag-interview-prep
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env          # optional – add an API key for LLM mode
uvicorn app.main:app --reload
```

Open **http://localhost:8000**. The index builds automatically on first start.

### Enable LLM mode (optional)

Put one key in `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...      # model: ANTHROPIC_MODEL (default claude-sonnet-5-5)
# or
OPENAI_API_KEY=sk-...             # model: OPENAI_MODEL (default gpt-4o-mini)
```

The header pill in the UI shows which mode you are in.

### Better semantic search (optional)

```bash
pip install sentence-transformers
```

`EMBEDDING_BACKEND=auto` will then use `all-MiniLM-L6-v2` (downloaded on first run) instead of TF-IDF. Re-index after switching (`POST /api/ingest` or the *Re-index* button).

### Docker

```bash
docker compose up --build
```

### Command line

```bash
python scripts/cli.py ingest
python scripts/cli.py ask "What is the GIL?" --topic python
python scripts/cli.py interview --topic machine-learning --n 3 --difficulty medium
```

---

## How it works

```
                 ┌───────────── ingestion ─────────────┐
 .md/.txt/.pdf → │ loaders → heading-aware chunker →   │ → data/index/
 .docx           │ embedder (TF-IDF or MiniLM) + BM25  │   (vectors.npy, chunks.json)
                 └─────────────────────────────────────┘

 question ─► dense search ┐
                          ├─► Reciprocal Rank Fusion ─► top-k chunks ─► LLM (cited answer)
           BM25 keywords ─┘                                         └► or extractive fallback

 mock interview:  question bank / LLM-generated question
                  → your answer → LLM grader or heuristic grader → feedback + summary
```

| Piece | File | Notes |
|---|---|---|
| Loading | `app/loaders.py` | Topic = sub-folder name, or file name |
| Chunking | `app/chunking.py` | Splits on headings, ignores `#` inside code fences, overlaps long sections |
| Embeddings | `app/embeddings.py` | `tfidf` (offline) or `sentence-transformers` |
| Store | `app/vectorstore.py` | NumPy matrix + JSON metadata; swap for FAISS/Qdrant/pgvector at scale |
| Retrieval | `app/retriever.py` | Dense + BM25, fused with RRF, optional topic filter |
| LLMs | `app/llm.py` | Anthropic / OpenAI / none |
| Interview | `app/interview.py` | Sessions, question selection, grading, summary |
| API | `app/main.py` | FastAPI; serves `frontend/index.html` |

### Heuristic grader (offline mode)

Score = 60 % IDF-weighted coverage of the reference answer's key terms + 40 % embedding similarity, with a penalty for very short answers. The feedback lists which concepts you hit and which you missed. It rewards covering the right ideas; it can't judge reasoning quality the way an LLM can, so use LLM mode for nuanced grading.

---

## Add your own material

Drop files into `data/sample_docs/` (or upload them in the UI), then re-index. To feed the mock-interview question bank, use this heading format:

```markdown
## Networking

### Q: [medium] What happens when you type a URL into the browser?
DNS resolution, TCP handshake, TLS, HTTP request, server response, rendering...
```

`[easy]`, `[medium]` and `[hard]` set the difficulty. Documents without Q&A headings still work for *Ask* and for LLM-generated interview questions.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Status, chunk count, active LLM/embedding backend |
| GET | `/api/topics` | Topics with chunk and question counts |
| POST | `/api/ask` | `{question, topic?, top_k?}` → answer + sources |
| POST | `/api/interview/start` | `{topic?, difficulty, num_questions}` → session + first question |
| POST | `/api/interview/answer` | `{session_id, answer}` → feedback + next question or summary |
| GET | `/api/interview/{id}/summary` | Results so far |
| POST | `/api/upload` | multipart `file` + optional `topic`; re-indexes |
| POST | `/api/ingest` | Rebuild the index |

Interactive docs: **http://localhost:8000/docs**

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Tests run fully offline (TF-IDF, no LLM).

## Configuration

All settings are environment variables — see `.env.example` (chunk size, overlap, top-k, backends, upload limit).

## Known limitations / next steps

- Interview sessions are held in memory and are lost on restart. Persist them in SQLite or Redis for multi-user use.
- No authentication — add it before exposing the app publicly.
- Re-indexing rebuilds everything; fine for thousands of chunks, but use incremental updates for large corpora.
- Ideas: cross-encoder re-ranking, query rewriting, spaced-repetition scheduling of weak topics, voice answers.

## Project layout

```
rag-interview-prep/
├── app/            # backend (FastAPI + RAG engine)
├── frontend/       # single-page UI
├── data/sample_docs/   # 7 topic study guides
├── scripts/cli.py  # terminal interface
├── tests/          # pytest suite
├── Dockerfile · docker-compose.yml
└── requirements.txt · .env.example
```
