"""Tiny text helpers shared by BM25 retrieval and the offline answer grader."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the and or but if then else of to in on for with as by at from is are was were be
    been being it its this that these those you your i we they he she them his her our their
    what which who whom how why when where do does did done can could should would will may
    might must not no yes so than too very just about into over under between within without
    also such each any all both more most other some only own same up down out off again
    further once here there""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def stem(word: str) -> str:
    """Very light suffix stripping — consistent, not linguistically perfect."""
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def analyze(text: str) -> list[str]:
    """Lowercase, drop stopwords, stem."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "general"
