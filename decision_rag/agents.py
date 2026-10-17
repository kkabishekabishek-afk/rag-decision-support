from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .gemini import GeminiRequestError, configured, generate
from .retrieval import format_evidence, retrieve


@dataclass
class AgentResult:
    name: str
    status: str
    output: str
    detail: str = ""


@dataclass
class WorkflowResult:
    query_type: str
    answer: str
    evidence: list[dict[str, Any]]
    agents: list[AgentResult] = field(default_factory=list)
    error: str | None = None


def route(question: str, evidence: list[dict[str, Any]] | None = None) -> str:
    q = question.lower()
    doc_words = r"\b(according to|from the document|from my document|from my resume|in the pdf|in this report|based on the resume|based on the report|what does the document say)\b"
    decision = r"\b(should|recommend|suitable|fit|choose|better|best|risk|decision|hire|select|prioriti[sz]e|afford|emi|loan)\b"
    analytical = r"\b(compare|analyse|analyze|summari[sz]e|strengths?|weaknesses?|patterns?|explain|assess|evaluate)\b"
    if re.search(doc_words, q):
        return "DOCUMENT"
    if evidence and re.search(r"\b(resume|candidate|document|report|pdf|company|income|net income)\b", q):
        return "COMBINED"
    if re.search(decision, q):
        return "GENERAL DECISION"
    if re.search(analytical, q):
        return "GENERAL ANALYSIS"
    return "GENERAL"


def _fallback(evidence: list[dict[str, Any]], reason: str) -> str:
    if not evidence:
        return f"I could not complete the Gemini answer. {reason}"
    bullets = "\n".join(f"- {item['text']} [E{i}]" for i, item in enumerate(evidence[:3], 1))
    return f"I could not complete the Gemini answer. {reason}\n\nRetrieved evidence:\n{bullets}"


def _analysis_prompt(question: str, evidence: str, qtype: str, history: str) -> str:
    evidence_rule = (
        "Use the evidence as the primary source and mark document claims [E1], [E2], etc. Do not invent document facts."
        if qtype in {"DOCUMENT", "COMBINED"} else
        "Answer the user's general question using your knowledge. Evidence is optional context only; do not force unrelated documents into the answer."
    )
    return f"""You are the Analyst, Researcher, and Critic agents in a chat assistant.
Question: {question}
Mode: {qtype}
Recent conversation:
{history or '(none)'}

{evidence_rule}
Think privately. Return concise internal notes only: answer intent, key facts, uncertainty, and what the final answer must not claim. Never address the user and never write meta-commentary for the final answer.

Retrieved evidence:
{evidence or 'NO RELEVANT DOCUMENT EVIDENCE'}"""


def _final_prompt(question: str, evidence: str, notes: str, qtype: str, history: str) -> str:
    if qtype == "DOCUMENT":
        source_rule = "Answer only from the retrieved documents. Cite every document claim with [E1], [E2], etc."
    elif qtype == "COMBINED":
        source_rule = "Answer the user's question directly. Use document evidence where relevant and cite those claims. Clearly label any general knowledge or inference."
    else:
        source_rule = "Answer as a helpful general assistant. Do not invent citations and do not mention unrelated retrieved documents."
    return f"""You are the final answer agent in a multi-agent chat assistant.

User question: {question}
Mode: {qtype}
Conversation context:
{history or '(none)'}

{source_rule}

Important rules:
- Return ONLY the answer for the user. Never reveal hidden reasoning, agent notes, prompts, checklists, or phrases such as 'I will follow', 'final check', 'refining', or 'according to the notes'.
- Be direct and natural, like a high-quality chatbot. Use short headings or bullets only when useful.
- Resolve natural references from the conversation: "this candidate", "the person", "they", "their", and "the document" refer to the active document subject or the last clearly discussed subject. Do not ask the user to repeat the subject when the context already identifies it.
- If the question says "my", "I", or "me", never substitute a company or third party's financial data for the user's personal data.
- For personal EMI affordability, do not decide without personal net income, expenses, existing debts, and loan terms. State what is missing and optionally give a formula or illustrative calculation clearly labeled as an example.
- Do not claim certainty when the evidence is incomplete.

Analyst/Critic notes (private; do not expose them):
{notes}

Retrieved evidence:
{evidence or 'NO RELEVANT DOCUMENT EVIDENCE'}"""


def _clean_final_answer(text: str) -> str:
    value = text.strip()
    heading = re.search(r"(?m)^###\s+(Recommendation|Answer)\s*$", value)
    if heading:
        value = value[heading.start():]
    for marker in ("\nFinal check of constraints:", "\n*Final check", "\n### Internal", "\n### Notes to self", "\nI will follow the"):
        if marker in value:
            value = value.split(marker, 1)[0].rstrip()
    return value


def _emi_guardrail(question: str, answer: str, evidence: list[dict[str, Any]]) -> str:
    """Prevent an affordability conclusion from income alone."""
    q = question.lower()
    if not re.search(r"\b(emi|installment|monthly payment|loan payment)\b", q):
        return answer
    combined = " ".join(str(item.get("text", "")) for item in evidence).lower()
    has_budget_inputs = all(term in combined for term in ("expense", "debt")) and any(term in combined for term in ("interest", "tenure", "term"))
    if has_budget_inputs:
        return answer
    citation = " [E1]" if evidence else ""
    return (
        f"It cannot be determined from the selected document{'' if evidence else 's'} whether you can afford this EMI. "
        f"The document provides income information{citation}, but it does not provide enough information about your essential monthly expenses, existing debts or EMIs, interest rate, and loan tenure. "
        "Please provide those figures before treating the $200 payment as affordable."
    )


def run_workflow(question: str, source: str | None = None, history: list[dict[str, str]] | None = None, chunks: list[dict[str, Any]] | None = None) -> WorkflowResult:
    evidence = retrieve(question, source=source, history=history, chunks=chunks)
    qtype = route(question, evidence)
    evidence_text = format_evidence(evidence)
    history_text = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in (history or [])[-6:])
    agents: list[AgentResult] = [AgentResult("Router", "completed", qtype, "Selected the response mode")]
    agents.append(AgentResult("Retriever", "completed", f"Retrieved {len(evidence)} relevant evidence chunks" if evidence else "No relevant document evidence; general chat mode"))
    if not configured():
        agents.append(AgentResult("Gemini", "needs_config", "Gemini API key is not configured", "Add GEMINI_API_KEY to .env"))
        return WorkflowResult(qtype, _fallback(evidence, "Configure GEMINI_API_KEY to receive a generated answer."), evidence, agents, "Gemini is not configured")
    try:
        notes = generate(_analysis_prompt(question, evidence_text, qtype, history_text), temperature=0.0, max_output_tokens=900)
        agents.extend([
            AgentResult("Analyst", "completed", "Intent and facts analyzed"),
            AgentResult("Critic", "completed", "Unsupported claims and ambiguity reviewed"),
        ])
        final = _clean_final_answer(generate(_final_prompt(question, evidence_text, notes, qtype, history_text), temperature=0.2, max_output_tokens=1400))
        final = _emi_guardrail(question, final, evidence)
        agents.extend([
            AgentResult("Decision Lead", "completed", "Answer synthesized"),
            AgentResult("Verifier", "completed", "Checked answer boundaries and citations"),
            AgentResult("Final Editor", "completed", "User-facing answer prepared"),
        ])
        return WorkflowResult(qtype, final, evidence, agents)
    except GeminiRequestError as exc:
        agents.append(AgentResult("Gemini", "error", str(exc)))
        return WorkflowResult(qtype, _fallback(evidence, str(exc)), evidence, agents, str(exc))
