"""
DocRAG — LLM-Powered Document Q&A System
Streamlit UI entry point
"""

import streamlit as st
import os
from pathlib import Path
from ingest import ingest_document, list_documents
from utils.retriever import retrieve_context
from utils.llm import ask_llm
from utils.helpers import format_sources

st.set_page_config(
    page_title="DocRAG — Document Q&A",
    page_icon="⬡",
    layout="wide",
)

# ── Styles ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #0a0a0b; }
    .stApp { background: #0a0a0b; color: #f0ede8; }
    .source-card {
        background: #18181c;
        border-left: 3px solid #e8c547;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 2px;
        font-size: 13px;
    }
    .score-badge {
        background: rgba(232,197,71,0.15);
        color: #e8c547;
        padding: 2px 8px;
        font-size: 11px;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_docs" not in st.session_state:
    st.session_state.active_docs = []

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⬡ DocRAG")
    st.caption("Retrieval-Augmented Generation Q&A")
    st.divider()

    # File upload
    st.markdown("### 📂 Upload Document")
    uploaded_file = st.file_uploader(
        "PDF or DOCX",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_file:
        for f in uploaded_file:
            with st.spinner(f"Ingesting {f.name}…"):
                doc_id = ingest_document(f)
                if doc_id and doc_id not in st.session_state.active_docs:
                    st.session_state.active_docs.append(doc_id)
            st.success(f"✓ {f.name} indexed")

    st.divider()

    # Document list
    st.markdown("### 📄 Documents")
    all_docs = list_documents()
    if not all_docs:
        st.caption("No documents yet — upload one above.")
    else:
        for doc in all_docs:
            checked = doc["id"] in st.session_state.active_docs
            if st.checkbox(doc["name"], value=checked, key=f"doc_{doc['id']}"):
                if doc["id"] not in st.session_state.active_docs:
                    st.session_state.active_docs.append(doc["id"])
            else:
                st.session_state.active_docs = [
                    d for d in st.session_state.active_docs if d != doc["id"]
                ]
            st.caption(f"  {doc['chunks']} chunks · {doc['size']}")

    st.divider()
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ── Main chat area ───────────────────────────────────────────────────────────
st.markdown("## Ask your documents")

if not st.session_state.active_docs:
    st.info("⬡ Upload and select a document from the sidebar to begin.")
else:
    # Render message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Retrieved Sources", expanded=False):
                    st.markdown(format_sources(msg["sources"]), unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents…"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve + generate
        with st.chat_message("assistant"):
            with st.spinner("Searching documents…"):
                context_chunks = retrieve_context(
                    query=prompt,
                    doc_ids=st.session_state.active_docs,
                    top_k=4,
                )
            with st.spinner("Generating answer…"):
                answer, sources = ask_llm(
                    question=prompt,
                    context_chunks=context_chunks,
                    history=st.session_state.messages[:-1],
                )
            st.markdown(answer)
            if sources:
                with st.expander("📎 Retrieved Sources", expanded=True):
                    st.markdown(format_sources(sources), unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
