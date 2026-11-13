"""FastAPI application."""
from __future__ import annotations

import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import BASE_DIR, Settings
from .loaders import SUPPORTED_EXTENSIONS
from .service import RAGService
from .textutils import slugify

FRONTEND = BASE_DIR / "frontend" / "index.html"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    topic: str | None = None
    top_k: int = Field(5, ge=1, le=15)


class StartRequest(BaseModel):
    topic: str | None = None
    difficulty: Literal["easy", "medium", "hard", "mixed"] = "mixed"
    num_questions: int = Field(5, ge=1, le=20)


class AnswerRequest(BaseModel):
    session_id: str
    answer: str = Field(min_length=1, max_length=6000)


def create_app(settings: Settings | None = None) -> FastAPI:
    service = RAGService(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service.ensure_index()
        yield

    app = FastAPI(title="RAG Interview Prep", version=__version__, lifespan=lifespan)
    app.state.service = service

    def guard(fn):
        """Map domain errors to HTTP errors."""
        try:
            return fn()
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'\""))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/", include_in_schema=False)
    def index():
        if not FRONTEND.exists():
            raise HTTPException(404, "frontend/index.html not found")
        return FileResponse(FRONTEND)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__, **service.stats()}

    @app.get("/api/topics")
    def topics():
        return {"topics": service.topics()}

    @app.post("/api/ask")
    def ask(req: AskRequest):
        return guard(lambda: service.ask(req.question, req.topic, req.top_k))

    @app.post("/api/interview/start")
    def interview_start(req: StartRequest):
        def run():
            topic = service.normalise_topic(req.topic)
            session, question = service.engine.start(topic, req.difficulty, req.num_questions)
            return {"session_id": session.id, "topic": topic or "all", "question": question}

        return guard(run)

    @app.post("/api/interview/answer")
    def interview_answer(req: AnswerRequest):
        return guard(lambda: service.engine.submit_answer(req.session_id, req.answer))

    @app.get("/api/interview/{session_id}/summary")
    def interview_summary(session_id: str):
        return guard(lambda: service.engine.summary(service.engine.get(session_id)))

    @app.post("/api/ingest")
    def ingest():
        return guard(service.build_index)

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...), topic: str = Form("")):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type '{suffix}'. Use: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        data = await file.read()
        if len(data) > service.settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"File exceeds {service.settings.max_upload_mb} MB")

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(file.filename).name)
        folder = service.settings.upload_dir / slugify(topic) if topic.strip() else service.settings.upload_dir
        folder.mkdir(parents=True, exist_ok=True)
        (folder / safe_name).write_bytes(data)
        stats = guard(service.build_index)
        return {"saved": str(Path(folder.name if topic.strip() else "") / safe_name), **stats}

    return app


app = create_app()
