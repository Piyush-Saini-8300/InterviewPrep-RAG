"""Pluggable embedding backends.

* `sentence-transformers` — real semantic embeddings (optional dependency).
* `tfidf` — scikit-learn TF-IDF with uni+bi-grams. Works fully offline, no model download.

Every backend returns L2-normalised float32 vectors so cosine similarity is a dot product.
"""
from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np

from .config import Settings


class Embedder:
    name = "base"

    def fit(self, texts: list[str]) -> None:  # noqa: D401 - no-op for pretrained models
        """Learn corpus statistics if the backend needs them."""

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def save(self, directory: Path) -> None:
        raise NotImplementedError


class TfidfEmbedder(Embedder):
    name = "tfidf"

    def __init__(self, vectorizer=None, max_features: int = 8000):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = vectorizer or TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=max_features,
        )

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray().astype("float32")

    def save(self, directory: Path) -> None:
        with open(directory / "tfidf.pkl", "wb") as fh:
            pickle.dump(self.vectorizer, fh)

    @classmethod
    def load(cls, directory: Path) -> "TfidfEmbedder":
        with open(directory / "tfidf.pkl", "rb") as fh:
            return cls(vectorizer=pickle.load(fh))


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

    def save(self, directory: Path) -> None:
        (directory / "st_model.json").write_text(json.dumps({"model": self.model_name}))

    @classmethod
    def load(cls, directory: Path) -> "SentenceTransformerEmbedder":
        meta = json.loads((directory / "st_model.json").read_text())
        return cls(meta["model"])


def make_embedder(settings: Settings) -> Embedder:
    backend = settings.embedding_backend
    if backend in ("auto", "sentence-transformers"):
        if importlib.util.find_spec("sentence_transformers") is not None:
            try:
                return SentenceTransformerEmbedder(settings.embedding_model)
            except Exception as exc:
                if backend == "sentence-transformers":
                    raise
                print(f"[embeddings] could not load '{settings.embedding_model}' ({exc}); using TF-IDF")
        elif backend == "sentence-transformers":
            raise RuntimeError("sentence-transformers is not installed (pip install sentence-transformers)")
    return TfidfEmbedder()


def load_embedder(directory: Path, backend: str) -> Embedder:
    if backend == SentenceTransformerEmbedder.name:
        return SentenceTransformerEmbedder.load(directory)
    return TfidfEmbedder.load(directory)
