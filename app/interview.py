"""Mock-interview engine: question generation, answer grading and session summaries.

With an LLM configured, questions are generated from retrieved study material and answers are
graded against it. Without one, the engine falls back to the "### Q:" question bank in your docs
and a transparent keyword + semantic-similarity grader, so it always works offline.
"""
from __future__ import annotations

import random
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from statistics import mean

import numpy as np

from . import prompts
from .chunking import Chunk
from .llm import LLM, extract_json
from .retriever import HybridRetriever
from .textutils import STOPWORDS, analyze, stem, tokenize

DIFFICULTIES = ("easy", "medium", "hard", "mixed")


def verdict_for(score: float) -> str:
    if score >= 8:
        return "Strong"
    if score >= 6:
        return "Good"
    if score >= 4:
        return "Partial"
    return "Needs work"


@dataclass
class Turn:
    question: str
    reference: str
    sources: list[dict]
    difficulty: str
    chunk_ids: list[str]
    answer: str | None = None
    score: float | None = None
    feedback: dict | None = None


@dataclass
class Session:
    id: str
    topic: str | None
    difficulty: str
    total: int
    turns: list[Turn] = field(default_factory=list)
    finished: bool = False
    created: float = field(default_factory=time.time)

    @property
    def current(self) -> Turn | None:
        if self.turns and self.turns[-1].answer is None:
            return self.turns[-1]
        return None

    @property
    def asked(self) -> list[str]:
        return [t.question for t in self.turns]


class InterviewEngine:
    def __init__(self, retriever: HybridRetriever, llm: LLM, rng: random.Random | None = None):
        self.retriever = retriever
        self.llm = llm
        self.rng = rng or random.Random()
        self.sessions: dict[str, Session] = {}

    def set_retriever(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    # ------------------------------------------------------------------ public API
    def start(self, topic: str | None, difficulty: str, num_questions: int) -> tuple[Session, dict]:
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
        if not np.any(self.retriever.store.mask_for_topic(topic)):
            raise ValueError(f"No study material found for topic '{topic}'")
        session = Session(uuid.uuid4().hex[:12], topic, difficulty, num_questions)
        self.sessions[session.id] = session
        return session, self._add_turn(session)

    def get(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            raise KeyError(f"Unknown session '{session_id}'")
        return self.sessions[session_id]

    def submit_answer(self, session_id: str, answer: str) -> dict:
        session = self.get(session_id)
        turn = session.current
        if session.finished or turn is None:
            raise ValueError("This interview is already finished")

        feedback = self._evaluate(turn, answer)
        turn.answer = answer
        turn.score = feedback["score"]
        turn.feedback = feedback

        result: dict = {"feedback": feedback, "finished": False, "next_question": None, "summary": None}
        if len(session.turns) >= session.total:
            session.finished = True
            result["finished"] = True
            result["summary"] = self.summary(session)
        else:
            result["next_question"] = self._add_turn(session)
        return result

    # ------------------------------------------------------------------ questions
    def _add_turn(self, session: Session) -> dict:
        turn = self._llm_question(session) if self.llm.available else None
        turn = turn or self._bank_question(session)
        session.turns.append(turn)
        return {
            "index": len(session.turns),
            "total": session.total,
            "question": turn.question,
            "difficulty": turn.difficulty,
            "sources": turn.sources,
        }

    def _pool(self, session: Session) -> list[Chunk]:
        mask = self.retriever.store.mask_for_topic(session.topic)
        chunks = self.retriever.store.chunks
        return [c for c, ok in zip(chunks, mask) if ok]

    def _used_ids(self, session: Session) -> set[str]:
        return {cid for t in session.turns for cid in t.chunk_ids}

    def _bank_question(self, session: Session) -> Turn:
        """Pick from the '### Q:' bank; otherwise turn a heading into a prompt."""
        used = self._used_ids(session)
        pool = [c for c in self._pool(session) if c.id not in used]
        if not pool:  # exhausted: allow repeats rather than failing
            pool = self._pool(session)

        questions = [c for c in pool if c.is_question]
        if session.difficulty != "mixed":
            exact = [c for c in questions if c.difficulty == session.difficulty]
            questions = exact or questions

        if questions:
            chunk = self.rng.choice(questions)
            question = chunk.question
        else:
            chunk = self.rng.choice(pool)
            subject = chunk.heading.split(">")[-1].strip()
            question = f"Explain {subject} and why it matters in practice."
        return Turn(
            question=question,
            reference=chunk.text,
            sources=[_chunk_source(chunk)],
            difficulty=chunk.difficulty if chunk.is_question else "medium",
            chunk_ids=[chunk.id],
        )

    def _llm_question(self, session: Session) -> Turn | None:
        try:
            used = self._used_ids(session)
            pool = [c for c in self._pool(session) if c.id not in used] or self._pool(session)
            seed = self.rng.choice(pool)
            hits = self.retriever.retrieve(f"{seed.heading} {seed.text[:200]}", k=3, topic=session.topic)
            if not any(h.chunk.id == seed.id for h in hits):
                hits = hits[:2]
            reference_chunks = [seed] + [h.chunk for h in hits if h.chunk.id != seed.id]
            context = "\n\n".join(f"[{i}] {c.heading}\n{c.text}" for i, c in enumerate(reference_chunks, 1))
            level = session.difficulty if session.difficulty != "mixed" else self.rng.choice(("easy", "medium", "hard"))
            raw = self.llm.generate(
                prompts.QUESTION_SYSTEM,
                prompts.question_user_prompt(context, level, session.asked),
                max_tokens=400,
                temperature=0.7,
            )
            data = extract_json(raw)
            question = (data or {}).get("question", "").strip()
            if not question:
                return None
            return Turn(
                question=question,
                reference="\n\n".join(c.text for c in reference_chunks),
                sources=[_chunk_source(c) for c in reference_chunks],
                difficulty=level,
                chunk_ids=[c.id for c in reference_chunks],
            )
        except Exception as exc:
            print(f"[interview] LLM question generation failed, using question bank: {exc}")
            return None

    # ------------------------------------------------------------------ grading
    def _evaluate(self, turn: Turn, answer: str) -> dict:
        if self.llm.available:
            try:
                raw = self.llm.generate(
                    prompts.EVAL_SYSTEM,
                    prompts.eval_user_prompt(turn.question, turn.reference, answer),
                    max_tokens=700,
                    temperature=0.2,
                )
                data = extract_json(raw)
                if data and "score" in data:
                    score = max(0.0, min(10.0, float(data["score"])))
                    return {
                        "score": round(score, 1),
                        "verdict": verdict_for(score),
                        "strengths": _as_list(data.get("strengths")),
                        "improvements": _as_list(data.get("improvements")),
                        "missing_concepts": _as_list(data.get("missing_concepts")),
                        "model_answer": str(data.get("model_answer") or turn.reference),
                        "method": "llm",
                    }
            except Exception as exc:
                print(f"[interview] LLM grading failed, using heuristic grader: {exc}")
        return self._heuristic_eval(turn, answer)

    def _heuristic_eval(self, turn: Turn, answer: str) -> dict:
        """Keyword coverage (IDF-weighted) blended with embedding similarity."""
        idf = self.retriever.bm25.idf
        surface: dict[str, str] = {}
        counts: Counter[str] = Counter()
        for tok in tokenize(turn.reference):
            if tok in STOPWORDS or len(tok) < 4:
                continue
            s = stem(tok)
            counts[s] += 1
            surface.setdefault(s, tok)

        weights = {t: c * idf.get(t, 1.0) for t, c in counts.items()}
        key_terms = sorted(weights, key=weights.get, reverse=True)[:14]
        total_weight = sum(weights[t] for t in key_terms) or 1.0

        answer_terms = set(analyze(answer))
        hit = [t for t in key_terms if t in answer_terms]
        missed = [t for t in key_terms if t not in answer_terms]
        coverage = sum(weights[t] for t in hit) / total_weight

        embedder = self.retriever.embedder
        a_vec, r_vec = embedder.encode([answer])[0], embedder.encode([turn.reference])[0]
        denom = float(np.linalg.norm(a_vec) * np.linalg.norm(r_vec)) or 1.0
        similarity = max(0.0, float(a_vec @ r_vec) / denom)
        semantic = min(1.0, similarity / 0.45)

        score = 10 * (0.6 * coverage + 0.4 * semantic)
        n_words = len(answer.split())
        if n_words < 8:
            score *= 0.4
        elif n_words < 20:
            score *= 0.75
        score = round(max(0.0, min(10.0, score)), 1)

        strengths = []
        if hit:
            strengths.append("You covered key ideas such as: " + ", ".join(surface[t] for t in hit[:6]) + ".")
        if n_words >= 40:
            strengths.append("Your answer had good depth and detail.")
        improvements = []
        if missed:
            improvements.append("Try to also mention: " + ", ".join(surface[t] for t in missed[:6]) + ".")
        if n_words < 20:
            improvements.append("Give a fuller answer — define the idea, explain how it works, and add an example.")
        if not strengths:
            strengths.append("Keep going — review the model answer below and try again.")

        return {
            "score": score,
            "verdict": verdict_for(score),
            "strengths": strengths,
            "improvements": improvements,
            "missing_concepts": [surface[t] for t in missed[:8]],
            "model_answer": turn.reference,
            "method": "heuristic",
        }

    # ------------------------------------------------------------------ summary
    def summary(self, session: Session) -> dict:
        scored = [t for t in session.turns if t.score is not None]
        average = round(mean(t.score for t in scored), 1) if scored else 0.0
        rows = [
            {"question": t.question, "score": t.score, "verdict": verdict_for(t.score), "difficulty": t.difficulty}
            for t in scored
        ]
        gaps = Counter(c for t in scored for c in (t.feedback or {}).get("missing_concepts", []))
        weakest = sorted(rows, key=lambda r: r["score"])[:2]

        readiness = (
            "Interview-ready" if average >= 8 else
            "Nearly there" if average >= 6 else
            "Keep practising" if average >= 4 else
            "Start with the fundamentals"
        )
        coach = None
        if self.llm.available and rows:
            try:
                coach = self.llm.generate(
                    prompts.COACH_SYSTEM,
                    prompts.coach_user_prompt(session.topic or "mixed topics", rows),
                    max_tokens=300,
                    temperature=0.4,
                )
            except Exception as exc:
                print(f"[interview] coach summary failed: {exc}")

        return {
            "session_id": session.id,
            "topic": session.topic or "all topics",
            "average_score": average,
            "readiness": readiness,
            "questions": rows,
            "weakest": weakest,
            "top_gaps": [term for term, _ in gaps.most_common(8)],
            "coach": coach,
        }


def _chunk_source(chunk: Chunk) -> dict:
    return {"id": chunk.id, "source": chunk.source, "topic": chunk.topic, "heading": chunk.heading}


def _as_list(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value] if isinstance(value, list) else []
