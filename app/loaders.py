"""Load documents (md / txt / pdf / docx) from one or more directories.

Topic = first sub-folder under the root, otherwise the file name (without extension).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .textutils import slugify

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf", ".docx"}


@dataclass
class Document:
    source: str  # path relative to its root, e.g. "python.md"
    topic: str  # slug, e.g. "python"
    text: str
    mtime: float = 0.0


def infer_topic(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return slugify(rel.parts[0] if len(rel.parts) > 1 else path.stem)


def read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".docx":
        import docx  # python-docx

        document = docx.Document(str(path))
        lines = []
        for para in document.paragraphs:
            style = (para.style.name or "").lower() if para.style is not None else ""
            if style.startswith("heading") and para.text.strip():
                digits = "".join(ch for ch in style if ch.isdigit())
                level = int(digits) if digits else 1
                lines.append("#" * min(level, 6) + " " + para.text.strip())
            else:
                lines.append(para.text)
        return "\n\n".join(lines)
    raise ValueError(f"Unsupported file type: {suffix}")


def load_documents(roots: list[Path]) -> list[Document]:
    docs: list[Document] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if path.name.startswith("."):
                continue
            try:
                text = read_text(path)
            except Exception as exc:  # corrupt PDF, bad encoding, ...
                print(f"[loaders] skipped {path}: {exc}")
                continue
            if not text.strip():
                continue
            docs.append(
                Document(
                    source=str(path.relative_to(root)),
                    topic=infer_topic(path, root),
                    text=text,
                    mtime=path.stat().st_mtime,
                )
            )
    return docs
