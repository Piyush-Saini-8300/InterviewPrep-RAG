"""A small, dependency-light vector store: a normalised matrix + chunk metadata on disk.

Perfect for thousands of chunks. For millions, swap this class for FAISS / Qdrant / pgvector —
the retriever only needs `vectors`, `chunks` and `mask_for_topic`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .chunking import Chunk


class VectorStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, meta: dict | None = None):
        assert len(chunks) == len(vectors), "chunks and vectors must align"
        self.chunks = chunks
        self.vectors = vectors.astype("float32")
        self.meta = meta or {}

    def __len__(self) -> int:
        return len(self.chunks)

    def mask_for_topic(self, topic: str | None) -> np.ndarray:
        if not topic:
            return np.ones(len(self.chunks), dtype=bool)
        return np.array([c.topic == topic for c in self.chunks], dtype=bool)

    def topics(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for c in self.chunks:
            info = out.setdefault(c.topic, {"chunks": 0, "questions": 0})
            info["chunks"] += 1
            info["questions"] += int(c.is_question)
        return out

    # ---- persistence -------------------------------------------------
    def save(self, directory: Path, extra_meta: dict | None = None) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False), encoding="utf-8"
        )
        meta = {**self.meta, **(extra_meta or {}), "n_chunks": len(self.chunks), "built_at": time.time()}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.meta = meta

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        chunks = [
            Chunk.from_dict(d)
            for d in json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        ]
        vectors = np.load(directory / "vectors.npy")
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        return cls(chunks, vectors, meta)

    @staticmethod
    def exists(directory: Path) -> bool:
        return all((directory / n).exists() for n in ("vectors.npy", "chunks.json", "meta.json"))
