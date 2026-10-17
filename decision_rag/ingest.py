from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "documents"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9_\-]{1,}", text.lower())


def _chunk(text: str, size: int = 1100, overlap: int = 160) -> list[str]:
    words = re.findall(r"\S+", text)
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        part = " ".join(words[start : start + size]).strip()
        if part:
            chunks.append(part)
        if start + size >= len(words):
            break
    return chunks


def extract_pdf(pdf_path: str | Path, source_name: str | None = None) -> dict[str, Any]:
    """Extract a PDF into memory only. No PDF, index, or cache file is written."""
    path = Path(pdf_path)
    name = source_name or path.name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    reader = PdfReader(str(path))
    records: list[dict[str, Any]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\x00", " ").strip()
        for chunk_number, chunk in enumerate(_chunk(text), start=1):
            records.append({
                "id": hashlib.sha1(f"{digest}:{page_number}:{chunk_number}:{chunk}".encode()).hexdigest(),
                "source": name,
                "sha256": digest,
                "page": page_number,
                "chunk": chunk_number,
                "text": chunk,
                "terms": sorted(set(_tokens(chunk))),
            })
    if not records:
        raise ValueError("No extractable text was found in this PDF.")
    return {"source": name, "sha256": digest, "pages": len(reader.pages), "chunks": records}


def index_pdf(pdf_path: str | Path) -> dict[str, Any]:
    """Compatibility wrapper; extraction remains memory-only and writes nothing."""
    result = extract_pdf(pdf_path)
    return {"source": result["source"], "pages": result["pages"], "chunks": len(result["chunks"]), "cached": False}


def list_documents() -> list[dict[str, Any]]:
    """Persistent document libraries are intentionally disabled."""
    return []


def get_chunks(source: str | None = None) -> list[dict[str, Any]]:
    """The UI passes session chunks explicitly; no disk index is read."""
    return []
