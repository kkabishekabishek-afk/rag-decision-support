# Decision RAG — rebuild plan

## Goal
Create a reliable Streamlit decision-support application that uses retrieval-augmented generation (RAG) and a multi-agent pipeline backed by Gemini. It must work with uploaded PDFs, show evidence, avoid cross-document leakage, and fail with actionable messages when Gemini is not configured.

## Implementation approach
- **Document ingestion:** `pypdf` extracts page-aware text and creates deterministic overlapping chunks.
- **Retrieval:** a lightweight local TF-IDF-style lexical retriever ranks chunks without downloading a transformer or depending on a prebuilt Chroma database. Each query can be scoped to one document or all indexed documents.
- **Multi-agent workflow:** Router -> Retriever -> Analyst -> Risk -> Options -> Decision -> Verifier. Router uses deterministic rules; Gemini is used for reasoning agents when configured.
- **Gemini boundary:** one typed client, one model setting (`GEMINI_MODEL`), structured JSON responses where useful, timeout/error translation, and no Ollama compatibility layer.
- **Persistence:** indexed chunks are stored as JSON under `data/index.json`; PDFs remain under `data/documents`. The index is rebuilt safely with atomic writes.
- **UI:** single Streamlit app with upload/index controls, document selector, query box, agent trace, final answer, evidence citations, and a clear configuration panel.

## Project structure
- `app.py` — Streamlit UI and session state.
- `decision_rag/ingest.py` — PDF extraction, chunking, index persistence.
- `decision_rag/retrieval.py` — deterministic lexical retrieval and evidence formatting.
- `decision_rag/gemini.py` — Gemini API wrapper and safe JSON/text generation.
- `decision_rag/agents.py` — multi-agent orchestration and prompts.
- `requirements.txt`, `.env.example`, `README.md` — setup and operation.
- `data/documents/` — supplied and uploaded PDFs.

## Design
- **Movement:** editorial operations console: dark ink, warm paper, and signal green.
- **Principles:** evidence first, calm hierarchy, visible provenance, progressive disclosure.
- **Palette:** charcoal for trust, cream for readability, green for verified states, amber for uncertainty.
- **Layout:** wide two-column workspace; left rail for corpus and controls, main canvas for the decision brief.
- **Signature elements:** numbered agent rail, citation chips, and a “confidence is evidence coverage” status line.
- **Interaction:** every action explains what it changes; missing configuration is a setup state, not a crash.
- **Typography:** system sans for UI with compact uppercase labels and generous reading width for answers.
- **Brand essence:** evidence-backed decisions from the documents you already own; grounded, transparent, practical.
