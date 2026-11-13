"""RAGService: owns the index, retriever, LLM and interview engine."""
from __future__ import annotations

import threading
import time

from .chunking import chunk_documents
from .config import Settings
from .embeddings import load_embedder, make_embedder
from .interview import InterviewEngine
from .llm import make_llm
from .loaders import load_documents
from .prompts import QA_SYSTEM, qa_user_prompt
from .retriever import HybridRetriever
from .textutils import slugify
from .vectorstore import VectorStore


class RAGService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.llm = make_llm(self.settings)
        self.store: VectorStore | None = None
        self.embedder = None
        self.retriever: HybridRetriever | None = None
        self.engine: InterviewEngine | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ index lifecycle
    def build_index(self) -> dict:
        with self._lock:
            started = time.time()
            docs = load_documents(self.settings.doc_dirs)
            chunks = chunk_documents(
                docs,
                self.settings.chunk_words,
                self.settings.chunk_overlap_words,
                self.settings.min_chunk_words,
            )
            if not chunks:
                raise ValueError(
                    "No documents found. Add .md/.txt/.pdf/.docx files to data/sample_docs or upload some."
                )
            embedder = make_embedder(self.settings)
            texts = [c.index_text for c in chunks]
            embedder.fit(texts)
            store = VectorStore(chunks, embedder.encode(texts))
            store.save(self.settings.index_dir, {"embedding_backend": embedder.name, "documents": len(docs)})
            embedder.save(self.settings.index_dir)
            self._attach(store, embedder)
            return {**self.stats(), "seconds": round(time.time() - started, 2)}

    def load_index(self) -> bool:
        directory = self.settings.index_dir
        if not VectorStore.exists(directory):
            return False
        try:
            store = VectorStore.load(directory)
            embedder = load_embedder(directory, store.meta.get("embedding_backend", "tfidf"))
        except Exception as exc:
            print(f"[service] could not load saved index ({exc}); rebuilding")
            return False
        self._attach(store, embedder)
        return True

    def ensure_index(self) -> None:
        with self._lock:
            if self.load_index() and not self._is_stale():
                return
            self.build_index()

    def _is_stale(self) -> bool:
        built_at = (self.store.meta or {}).get("built_at", 0)
        newest = max((d.mtime for d in load_documents(self.settings.doc_dirs)), default=0)
        return newest > built_at

    def _attach(self, store: VectorStore, embedder) -> None:
        self.store, self.embedder = store, embedder
        self.retriever = HybridRetriever(store, embedder)
        if self.engine is None:
            self.engine = InterviewEngine(self.retriever, self.llm)
        else:
            self.engine.set_retriever(self.retriever)

    # ------------------------------------------------------------------ queries
    def stats(self) -> dict:
        assert self.store is not None
        return {
            "chunks": len(self.store),
            "topics": len(self.store.topics()),
            "documents": self.store.meta.get("documents"),
            "embedding": self.store.meta.get("embedding_backend"),
            "llm": self.llm.provider,
        }

    def topics(self) -> list[dict]:
        assert self.store is not None
        return [
            {"topic": t, "label": t.replace("-", " ").title(), **info}
            for t, info in sorted(self.store.topics().items())
        ]

    def normalise_topic(self, topic: str | None) -> str | None:
        if not topic or topic.lower() in {"all", "mixed", "any"}:
            return None
        slug = slugify(topic)
        if slug not in self.store.topics():
            raise ValueError(f"Unknown topic '{topic}'")
        return slug

    def ask(self, question: str, topic: str | None = None, top_k: int | None = None) -> dict:
        assert self.retriever is not None
        hits = self.retriever.retrieve(question, k=top_k or self.settings.top_k, topic=self.normalise_topic(topic))
        if not hits:
            return {
                "answer": "I couldn't find anything relevant in the knowledge base. "
                "Try rephrasing, or upload notes on this topic.",
                "mode": "no-results",
                "sources": [],
            }
        sources = [h.to_source() for h in hits]
        if self.llm.available:
            try:
                answer = self.llm.generate(QA_SYSTEM, qa_user_prompt(question, hits), max_tokens=900)
                return {"answer": answer, "mode": "llm", "sources": sources}
            except Exception as exc:
                print(f"[service] LLM call failed, returning extractive answer: {exc}")

        top = hits[:2]
        answer = "\n\n".join(f"**{h.chunk.heading}**\n{h.chunk.text.strip()}" for h in top)
        return {
            "answer": answer,
            "mode": "extractive",
            "note": "No LLM configured — showing the most relevant passages from your notes.",
            "sources": sources,
        }
