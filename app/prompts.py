"""Prompt templates. Built with plain string functions so JSON braces never need escaping."""
from __future__ import annotations

from .retriever import Retrieved

QA_SYSTEM = (
    "You are an interview-preparation tutor. Answer the candidate's question using ONLY the "
    "numbered study-material passages provided. Cite passages inline like [1] or [2]. "
    "If the passages do not contain the answer, say so plainly instead of guessing. "
    "Be clear and concise; use short code snippets where they help."
)

QUESTION_SYSTEM = (
    "You are a senior technical interviewer. From the study material provided, write ONE "
    "interview question that can be answered using only that material. Do not repeat any "
    "previously asked question. Respond with JSON only: "
    '{"question": "<the question>", "key_points": ["<point an ideal answer covers>", "..."]}'
)

EVAL_SYSTEM = (
    "You are a fair but rigorous interviewer grading a candidate's answer against the reference "
    "study material. Score 0-10 (10 = complete and accurate, 5 = partially correct, 0 = wrong or "
    "empty). Do not penalise different wording if the substance is right. Respond with JSON only: "
    '{"score": <number>, "strengths": ["..."], "improvements": ["..."], '
    '"missing_concepts": ["..."], "model_answer": "<a concise ideal answer>"}'
)

COACH_SYSTEM = (
    "You are an encouraging interview coach. Given a candidate's per-question scores, write a "
    "short (3-5 sentence) overall assessment: what they do well, the 2 most important gaps, and "
    "what to study next. Plain text, no markdown."
)


def format_context(hits: list[Retrieved], max_chars: int = 1600) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        text = hit.chunk.text.strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        blocks.append(f"[{i}] ({hit.chunk.source} — {hit.chunk.heading})\n{text}")
    return "\n\n".join(blocks)


def qa_user_prompt(question: str, hits: list[Retrieved]) -> str:
    return f"Study material:\n\n{format_context(hits)}\n\nQuestion: {question}\n\nAnswer:"


def question_user_prompt(context: str, difficulty: str, asked: list[str]) -> str:
    asked_block = "\n".join(f"- {q}" for q in asked) or "(none yet)"
    return (
        f"Difficulty: {difficulty}\n\nPreviously asked questions:\n{asked_block}\n\n"
        f"Study material:\n\n{context}"
    )


def eval_user_prompt(question: str, reference: str, answer: str) -> str:
    return (
        f"Interview question:\n{question}\n\nReference study material:\n{reference}\n\n"
        f"Candidate's answer:\n{answer}"
    )


def coach_user_prompt(topic: str, rows: list[dict]) -> str:
    lines = "\n".join(f"- ({r['score']}/10) {r['question']}" for r in rows)
    return f"Topic: {topic}\n\nResults:\n{lines}"
