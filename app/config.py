"""Central configuration, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader so we don't need python-dotenv."""
    path = path or BASE_DIR / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Settings:
    # Where documents live and where the built index is stored
    sample_dir: Path = BASE_DIR / "data" / "sample_docs"
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    index_dir: Path = BASE_DIR / "data" / "index"

    # Chunking
    chunk_words: int = 250
    chunk_overlap_words: int = 40
    min_chunk_words: int = 25

    # Retrieval
    top_k: int = 5

    # Embeddings: auto | sentence-transformers | tfidf
    embedding_backend: str = "auto"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM: auto | anthropic | openai | none
    llm_provider: str = "auto"
    anthropic_model: str = "claude-sonnet-5-5"
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    max_upload_mb: int = 20

    @property
    def doc_dirs(self) -> list[Path]:
        return [self.sample_dir, self.upload_dir]

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        env = os.environ.get
        return cls(
            chunk_words=int(env("CHUNK_WORDS", 250)),
            chunk_overlap_words=int(env("CHUNK_OVERLAP_WORDS", 40)),
            top_k=int(env("TOP_K", 5)),
            embedding_backend=env("EMBEDDING_BACKEND", "auto").lower(),
            embedding_model=env("EMBEDDING_MODEL", cls.embedding_model),
            llm_provider=env("LLM_PROVIDER", "auto").lower(),
            anthropic_model=env("ANTHROPIC_MODEL", cls.anthropic_model),
            openai_model=env("OPENAI_MODEL", cls.openai_model),
            anthropic_api_key=env("ANTHROPIC_API_KEY", ""),
            openai_api_key=env("OPENAI_API_KEY", ""),
            max_upload_mb=int(env("MAX_UPLOAD_MB", 20)),
        )
