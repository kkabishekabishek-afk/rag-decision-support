from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .ingest import get_chunks


REFERENCE_EXPANSIONS = {
    "this candidate": "candidate applicant person resume profile qualifications experience skills",
    "the candidate": "candidate applicant person resume profile qualifications experience skills",
    "this person": "person candidate applicant individual profile resume experience",
    "the person in the document": "person candidate applicant individual document profile resume",
    "they": "candidate applicant person individual",
    "their": "candidate applicant person individual",
    "he": "candidate applicant person individual",
    "she": "candidate applicant person individual",
    "it": "document report profile subject",
}


def _terms(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9_\-]{1,}", text.lower())


def expand_query(question: str, history: list[dict[str, str]] | None = None) -> str:
    """Add simple conversational aliases without changing the user's question."""
    expanded = question
    lower = question.lower()
    for phrase, aliases in REFERENCE_EXPANSIONS.items():
        if phrase in lower:
            expanded += " " + aliases
    if history and re.search(r"\b(this|that|they|their|he|she|the person|the candidate)\b", lower):
        previous_user = [m.get("content", "") for m in history if m.get("role") == "user"]
        if previous_user:
            expanded += " " + previous_user[-1]
    return expanded


def retrieve(question: str, source: str | None = None, top_k: int = 6, history: list[dict[str, str]] | None = None, chunks: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    candidates = chunks if chunks is not None else get_chunks(source)
    if source:
        candidates = [item for item in candidates if item.get("source") == source]
    if not candidates:
        return []
    query = Counter(_terms(expand_query(question, history)))
    if not query:
        return candidates[:top_k]
    doc_freq = Counter(term for item in candidates for term in set(item.get("terms", [])))
    scored: list[tuple[float, dict[str, Any]]] = []
    total = len(candidates)
    for item in candidates:
        terms = Counter(item.get("terms", []))
        score = 0.0
        for term, q_count in query.items():
            if terms[term]:
                idf = math.log((total + 1) / (doc_freq[term] + 1)) + 1
                score += idf * (1 + math.log(terms[term])) * (1 + math.log(q_count))
        if score:
            scored.append((score, item))
    scored.sort(key=lambda pair: (-pair[0], pair[1].get("source", ""), pair[1].get("page", 0)))
    return [{**item, "score": round(score, 4)} for score, item in scored[:top_k]]


def format_evidence(records: list[dict[str, Any]]) -> str:
    if not records:
        return "NO EVIDENCE RETRIEVED."
    return "\n\n".join(
        f"[E{i}] {item.get('source')} — page {item.get('page')}\n{item.get('text', '')}"
        for i, item in enumerate(records, start=1)
    )
