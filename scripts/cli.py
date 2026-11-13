#!/usr/bin/env python
"""Command-line interface.

    python scripts/cli.py ingest
    python scripts/cli.py ask "What is the GIL?" --topic python
    python scripts/cli.py interview --topic machine-learning --n 3 --difficulty medium
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.service import RAGService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Interview Prep CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="(Re)build the index from data/sample_docs and data/uploads")

    p_ask = sub.add_parser("ask", help="Ask a question against your notes")
    p_ask.add_argument("question")
    p_ask.add_argument("--topic")
    p_ask.add_argument("-k", type=int, default=5)

    p_int = sub.add_parser("interview", help="Run a mock interview in the terminal")
    p_int.add_argument("--topic")
    p_int.add_argument("--n", type=int, default=3)
    p_int.add_argument("--difficulty", default="mixed", choices=["easy", "medium", "hard", "mixed"])

    args = parser.parse_args()
    service = RAGService()

    if args.cmd == "ingest":
        print(service.build_index())
        return

    service.ensure_index()

    if args.cmd == "ask":
        result = service.ask(args.question, args.topic, args.k)
        print(result["answer"], "\n")
        for i, s in enumerate(result["sources"], 1):
            print(f"  [{i}] {s['source']} — {s['heading']}  (score {s['score']})")
        return

    topic = service.normalise_topic(args.topic)
    session, question = service.engine.start(topic, args.difficulty, args.n)
    print(f"\nMock interview — {topic or 'all topics'} — {args.n} questions (type 'quit' to stop)\n")
    while True:
        print(f"Q{question['index']}/{question['total']} [{question['difficulty']}]: {question['question']}")
        answer = input("> ").strip()
        if answer.lower() in {"quit", "exit"}:
            break
        result = service.engine.submit_answer(session.id, answer)
        fb = result["feedback"]
        print(f"\nScore: {fb['score']}/10 ({fb['verdict']})")
        for line in fb["strengths"]:
            print(f"  + {line}")
        for line in fb["improvements"]:
            print(f"  - {line}")
        print()
        if result["finished"]:
            s = result["summary"]
            print(f"Finished! Average {s['average_score']}/10 — {s['readiness']}")
            if s["top_gaps"]:
                print("Topics to revise:", ", ".join(s["top_gaps"]))
            if s["coach"]:
                print("\n" + s["coach"])
            break
        question = result["next_question"]


if __name__ == "__main__":
    main()
