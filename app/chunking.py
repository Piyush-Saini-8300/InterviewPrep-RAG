"""Heading-aware chunking.

Markdown is split on headings first (ignoring `#` inside code fences), so every
"### Q: ..." block stays together. Long sections are then packed by paragraph
with a small overlap. Question headings are tagged so the interview engine can
use them directly as a question bank:

    ### Q: [medium] What is the GIL?
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .loaders import Document
from .textutils import slugify

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
DIFFICULTY_RE = re.compile(r"\[(easy|medium|hard)\]", re.IGNORECASE)
QUESTION_RE = re.compile(r"^Q:\s*(.*)$", re.IGNORECASE)


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    topic: str
    heading: str
    is_question: bool = False
    question: str = ""
    difficulty: str = "medium"

    @property
    def index_text(self) -> str:
        """What gets embedded / BM25-indexed: heading gives the chunk its context."""
        return f"{self.heading}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


def split_sections(text: str) -> list[tuple[list[str], str]]:
    """Return [(heading_path, body)] — headings inside ``` fences are ignored."""
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []
    path: list[str] = []
    lines: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            sections.append((list(path), body))

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            lines.append(line)
            continue
        match = None if in_fence else HEADING_RE.match(line)
        if match:
            flush()
            lines = []
            level, title = len(match.group(1)), match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            path = [t for _, t in stack]
        else:
            lines.append(line)
    flush()
    return sections


def _hard_split(text: str, max_words: int, overlap: int) -> list[str]:
    multiline = "\n" in text
    sep = "\n" if multiline else " "
    units = text.splitlines() if multiline else text.split(" ")
    out: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for unit in units:
        w = max(1, len(unit.split()))
        if cur and cur_words + w > max_words:
            out.append(sep.join(cur))
            keep: list[str] = []
            kept = 0
            for prev in reversed(cur):
                pw = max(1, len(prev.split()))
                if kept + pw > overlap:
                    break
                keep.insert(0, prev)
                kept += pw
            cur, cur_words = keep, kept
        cur.append(unit)
        cur_words += w
    if cur:
        out.append(sep.join(cur))
    return out


def split_body(body: str, max_words: int, overlap: int) -> list[str]:
    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for para in paragraphs:
        w = len(para.split())
        if w > max_words:
            if cur:
                pieces.append("\n\n".join(cur))
                cur, cur_words = [], 0
            pieces.extend(_hard_split(para, max_words, overlap))
            continue
        if cur and cur_words + w > max_words:
            pieces.append("\n\n".join(cur))
            tail = cur[-1] if len(cur[-1].split()) <= overlap else None
            cur = [tail] if tail else []
            cur_words = len(tail.split()) if tail else 0
        cur.append(para)
        cur_words += w
    if cur:
        pieces.append("\n\n".join(cur))
    return pieces or [body]


def chunk_documents(
    docs: list[Document],
    chunk_words: int = 250,
    overlap: int = 40,
    min_words: int = 25,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        slug = slugify(f"{doc.topic}-{Path(doc.source).stem}")
        n = 0
        prev: Chunk | None = None
        for path, body in split_sections(doc.text):
            last = path[-1] if path else ""
            heading = " > ".join(path) if path else Path(doc.source).stem
            heading = re.sub(r"\s+", " ", DIFFICULTY_RE.sub("", heading)).strip()

            q_match = QUESTION_RE.match(last)
            question, difficulty = "", "medium"
            if q_match:
                question = q_match.group(1)
                d_match = DIFFICULTY_RE.search(question)
                if d_match:
                    difficulty = d_match.group(1).lower()
                question = re.sub(r"\s+", " ", DIFFICULTY_RE.sub("", question)).strip()

            pieces = split_body(body, chunk_words, overlap)

            # Fold tiny non-question sections into the previous chunk of the same doc
            if not q_match and prev is not None and len(pieces) == 1 and len(body.split()) < min_words:
                prev.text += "\n\n" + body
                continue

            for i, piece in enumerate(pieces):
                chunk = Chunk(
                    id=f"{slug}-{n}",
                    text=piece,
                    source=doc.source,
                    topic=doc.topic,
                    heading=heading,
                    is_question=bool(q_match) and i == 0,
                    question=question if i == 0 else "",
                    difficulty=difficulty,
                )
                n += 1
                chunks.append(chunk)
                prev = chunk
    return chunks
