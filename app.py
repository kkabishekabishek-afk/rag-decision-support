from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision_rag.agents import run_workflow
from decision_rag.gemini import configured, model_name
from decision_rag.ingest import extract_pdf

st.set_page_config(page_title="Decision RAG", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { --ink:#18221f; --muted:#5f6c66; --paper:#f7f8f6; --panel:#ffffff; --line:#dfe6e1; --green:#167a57; }
.stApp { background:var(--paper); color:var(--ink); }
.block-container { max-width:1100px; padding-top:1.5rem; padding-bottom:7rem; }
[data-testid="stSidebar"] { background:#f0f5f1; border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:var(--ink); }
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"], [data-testid="stChatMessage"] { color:var(--ink) !important; }
[data-baseweb="select"] *, textarea, input { color:var(--ink) !important; background:#fff !important; }
.hero { padding:.5rem 0 1.4rem; border-bottom:1px solid var(--line); margin-bottom:1.25rem; }
.eyebrow { color:var(--green); font-size:.7rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
.hero h1 { font-size:2.7rem; letter-spacing:-.055em; margin:.15rem 0 .3rem; }
.hero p { color:var(--muted); font-size:1.02rem; max-width:760px; }
.library-card { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:.7rem .85rem; margin:.4rem 0; }
.library-name { font-weight:650; overflow-wrap:anywhere; }
.library-meta { color:var(--muted); font-size:.78rem; margin-top:.15rem; }
[data-testid="stChatMessage"] { border:1px solid var(--line); border-radius:16px; margin:.5rem 0; padding:.35rem .75rem; background:var(--panel); }
[data-testid="stCaptionContainer"] { color:var(--muted) !important; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_chunks" not in st.session_state:
    st.session_state.session_chunks = []
if "session_docs" not in st.session_state:
    st.session_state.session_docs = []

st.markdown('<div class="hero"><div class="eyebrow">Temporary document chat</div><h1>Decision RAG</h1><p>Upload a PDF for this chat, ask questions naturally, and close the app when you are done. Uploaded files are processed in memory and are not saved as cache files.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Current chat files")
    docs = st.session_state.session_docs
    if docs:
        st.success(f"{len(docs)} file{'s' if len(docs) != 1 else ''} active in this chat")
        for doc in docs:
            st.markdown(f'<div class="library-card"><div class="library-name">{doc["source"]}</div><div class="library-meta">{doc["pages"]} pages · available until this chat ends</div></div>', unsafe_allow_html=True)
    else:
        st.info("No file attached to this chat")

    with st.expander("Attach PDF", expanded=not bool(docs)):
        uploads = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed", help="Files are read for this session only and are not copied to data/ or indexed on disk.")
        if uploads and st.button("Use in this chat", type="primary", use_container_width=True):
            added = 0
            for upload in uploads:
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary:
                        temporary.write(upload.getbuffer())
                        temp_path = Path(temporary.name)
                    extracted = extract_pdf(temp_path, source_name=upload.name)
                    st.session_state.session_chunks = [c for c in st.session_state.session_chunks if c.get("source") != upload.name] + extracted["chunks"]
                    st.session_state.session_docs = [d for d in st.session_state.session_docs if d.get("source") != upload.name] + [{"source": upload.name, "pages": extracted["pages"], "chunks": len(extracted["chunks"])}]
                    added += 1
                except Exception as exc:
                    st.error(f"Could not read {upload.name}: {exc}")
                finally:
                    if temp_path:
                        temp_path.unlink(missing_ok=True)
            if added:
                st.rerun()

    st.caption("Files are held only in Streamlit session memory. Restarting the app clears them.")
    st.divider()
    if configured():
        st.success(f"Gemini ready · {model_name()}")
    else:
        st.warning("Gemini needs configuration")
        st.caption("Add GEMINI_API_KEY to .env beside app.py.")
    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_chunks = []
        st.session_state.session_docs = []
        st.rerun()

scope_options = ["Use attached files"] + [d["source"] for d in docs]
scope = st.selectbox("Knowledge scope", scope_options, help="The assistant understands references like ‘this candidate’ and ‘the person in the document’ using your chat context.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        result = message.get("result")
        if result:
            st.caption(f"{result.query_type} · {len(result.evidence)} evidence sections")
            if result.evidence:
                with st.expander("View sources"):
                    for i, item in enumerate(result.evidence, 1):
                        st.markdown(f"**[E{i}] {item['source']} · page {item['page']}**")
                        st.write(item["text"])

prompt = st.chat_input("Message Decision RAG…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Reading context and preparing an answer…"):
            history = st.session_state.messages[:-1]
            selected = None if scope == "Use attached files" else scope
            result = run_workflow(prompt, selected, history=history, chunks=st.session_state.session_chunks)
        st.markdown(result.answer)
        st.caption(f"{result.query_type} · {len(result.evidence)} evidence sections")
        if result.error:
            st.warning(result.error)
        if result.evidence:
            with st.expander("View sources"):
                for i, item in enumerate(result.evidence, 1):
                    st.markdown(f"**[E{i}] {item['source']} · page {item['page']}**")
                    st.write(item["text"])
    st.session_state.messages.append({"role": "assistant", "content": result.answer, "result": result})
