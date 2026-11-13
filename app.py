"""Streamlit UI for Lumen Travel's sign-in Cần Thơ RAG assistant."""

import time
from typing import Any

import numpy as np
import streamlit as st

from firebase_client import FirebaseError, create_account, sign_in, validate_config
from rag import (
    DEFAULT_TOP_K,
    RAGError,
    SIMILARITY_THRESHOLD,
    answer_question,
    embed_texts,
    format_entry_for_embedding,
    load_knowledge_base,
)

SESSION_KEYS: tuple[str, ...] = (
    "user",
    "chat_history",
    "auth_email",
    "auth_password",
    "notice",
    "auth_error",
    "clear_password",
    "account_created",
)


def forget_user(state: dict[str, Any] | None = None) -> None:
    """Clear all per-user session state, including auth credentials and chat history."""
    target = st.session_state if state is None else state
    for key in SESSION_KEYS:
        target.pop(key, None)


@st.cache_resource
def get_indexed_guide() -> tuple[list[dict[str, Any]], np.ndarray]:
    """Load the Cần Thơ travel guide and compute embeddings once per server lifecycle."""
    entries = load_knowledge_base()
    texts = [format_entry_for_embedding(entry) for entry in entries]
    doc_vectors = embed_texts(texts)
    return entries, doc_vectors


def _render_retrieved_sources(retrieved: list[dict[str, Any]]) -> None:
    """Display retrieved guide entries and cosine similarity scores inside an expander."""
    with st.expander("Retrieved Guide Entries & Scores", expanded=False):
        for item in retrieved:
            entry = item["entry"]
            score = item["score"]
            st.markdown(f"**[{entry['id']}] {entry['title']}** — Score: `{score:.4f}`")
            if entry.get("category") or entry.get("tags"):
                tags = ", ".join(entry.get("tags", []))
                st.caption(f"Category: `{entry.get('category', 'general')}` | Tags: {tags}")
            st.write(entry["text"])
            st.caption(f"Source: {entry['source']}")
            st.divider()


def _render_auth_gate() -> None:
    """Render Firebase Email/Password sign-in and account creation form."""
    st.subheader("Sign in to access the Cần Thơ Travel Assistant")
    st.write("Use a separate app password (not your Google Account password).")

    try:
        with st.form("account_form", clear_on_submit=True):
            email = st.text_input("Email", key="auth_email")
            password = st.text_input("Password", type="password", key="auth_password")
            col1, col2 = st.columns(2)
            with col1:
                login = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            with col2:
                register = st.form_submit_button("Create account", use_container_width=True)

        if register:
            create_account(email, password)
            st.session_state["account_created"] = True
            st.session_state["clear_password"] = True
            st.rerun()
        elif login:
            user = sign_in(email, password)
            forget_user()
            st.session_state["user"] = user
            st.session_state["chat_history"] = []
            st.rerun()
    except FirebaseError as error:
        st.session_state["clear_password"] = True
        st.session_state["auth_error"] = str(error)
        st.rerun()


def _render_chat_page(user: dict[str, Any]) -> None:
    """Render the protected RAG chat interface for an authenticated user."""
    if time.time() >= float(user.get("expires_at", 0)):
        forget_user()
        st.session_state["notice"] = "Your session has expired. Please sign in again."
        st.rerun()

    header_col, action_col = st.columns([3, 1])
    with header_col:
        st.write(f"Signed in as **{user['email']}**")
        st.caption(
            f"Session-only memory (no database) · Top-{DEFAULT_TOP_K} cosine retrieval · "
            f"Min similarity threshold: {SIMILARITY_THRESHOLD:.2f}"
        )
    with action_col:
        if st.button("Sign out", use_container_width=True):
            forget_user()
            st.rerun()

    try:
        entries, doc_vectors = get_indexed_guide()
    except RAGError as error:
        st.error(str(error))
        if st.button("Retry Guide Indexing"):
            get_indexed_guide.clear()
            st.rerun()
        return

    chat_history: list[dict[str, Any]] = st.session_state.setdefault("chat_history", [])
    if not chat_history:
        st.info(
            "Ask anything about Cần Thơ — floating markets, heritage sites, orchards, local food, "
            "or river boat tips. Answers come strictly from our 16-entry guide."
        )

    for message in chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and message.get("retrieved"):
                _render_retrieved_sources(message["retrieved"])

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Ask a question about Cần Thơ",
            placeholder="e.g., What time does Cái Răng Floating Market open?",
        )
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Please enter a non-empty question.")
            return
        try:
            with st.spinner("Searching travel guide and generating answer..."):
                result = answer_question(question, entries, doc_vectors)
        except RAGError as error:
            st.error(str(error))
            return

        chat_history.append({"role": "user", "content": question.strip()})
        chat_history.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "retrieved": result["retrieved"],
            }
        )
        st.rerun()


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(page_title="Lumen Travel — Cần Thơ Assistant", layout="centered")
    st.title("Lumen Travel — Cần Thơ Assistant")
    st.caption("Firebase Authentication + Grounded Ollama RAG Chat (`embeddinggemma` + `qwen3:4b`)")

    try:
        validate_config()
    except FirebaseError as error:
        st.info(str(error))
        st.stop()

    if st.session_state.pop("clear_password", False):
        st.session_state["auth_password"] = ""
    if st.session_state.pop("account_created", False):
        st.success("Account created. Sign in with the same app credentials.")
    if "auth_error" in st.session_state:
        st.error(st.session_state.pop("auth_error"))
    if "notice" in st.session_state:
        st.info(st.session_state.pop("notice"))

    if "user" not in st.session_state:
        _render_auth_gate()
    else:
        _render_chat_page(st.session_state["user"])


if __name__ == "__main__":
    main()
