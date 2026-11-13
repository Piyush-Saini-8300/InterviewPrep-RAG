"""Hybrid retrieval: dense (embedding) + sparse (BM25), merged with Reciprocal Rank Fusion."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder
from .textutils import analyze
from .vectorstore import VectorStore

RRF_K = 60
POOL = 25  # candidates taken from each ranker before fusion


class BM25:
    """Okapi BM25 over an inverted index."""

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs_tokens)
        self.doc_len = np.array([len(d) for d in docs_tokens], dtype="float64")
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, tokens in enumerate(docs_tokens):
            for term, freq in Counter(tokens).items():
                self.postings[term].append((i, freq))
        self.idf = {
            term: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype="float64")
        if not self.n or not self.avgdl:
            return out
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freq in self.postings[term]:
                norm = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] += idf * freq * (self.k1 + 1) / norm
        return out


@dataclass
class Retrieved:
    chunk: Chunk
    score: float  # fused RRF score (higher = better)
    dense: float  # cosine similarity
    bm25: float

    def to_source(self, snippet_chars: int = 320) -> dict:
        text = " ".join(self.chunk.text.split())
        return {
            "id": self.chunk.id,
            "source": self.chunk.source,
            "topic": self.chunk.topic,
            "heading": self.chunk.heading,
            "score": round(self.score, 4),
            "dense": round(self.dense, 3),
            "snippet": text[:snippet_chars] + ("…" if len(text) > snippet_chars else ""),
        }


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self.bm25 = BM25([analyze(c.index_text) for c in store.chunks])

    def retrieve(self, query: str, k: int = 5, topic: str | None = None, mode: str = "hybrid") -> list[Retrieved]:
        candidates = np.flatnonzero(self.store.mask_for_topic(topic))
        if candidates.size == 0 or not query.strip():
            return []

        dense_all = self.store.vectors @ self.embedder.encode([query])[0]
        bm25_all = self.bm25.scores(analyze(query))

        def rank(scores: np.ndarray, floor: float) -> dict[int, int]:
            order = candidates[np.argsort(-scores[candidates])][:POOL]
            return {int(i): r for r, i in enumerate(order) if scores[i] > floor}

        dense_rank = rank(dense_all, 1e-6) if mode in ("hybrid", "dense") else {}
        bm25_rank = rank(bm25_all, 0.0) if mode in ("hybrid", "bm25") else {}

        fused: dict[int, float] = defaultdict(float)
        for ranks in (dense_rank, bm25_rank):
            for idx, r in ranks.items():
                fused[idx] += 1.0 / (RRF_K + r + 1)

        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            Retrieved(self.store.chunks[i], score, float(dense_all[i]), float(bm25_all[i]))
            for i, score in top
        ]
