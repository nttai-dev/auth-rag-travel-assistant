"""Two-layer RAG engine using Ollama embeddings (embeddinggemma) and chat (qwen3:4b)."""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import requests

from config import CHAT_MODEL, EMBED_MODEL, OLLAMA_BASE_URL

SIMILARITY_THRESHOLD = 0.35
DEFAULT_TOP_K = 3
IDK_RESPONSE = "I don't know based on the travel guide."

EMBED_TIMEOUT = 120
CHAT_TIMEOUT = 120
DEFAULT_KB_PATH = Path(__file__).with_name("knowledge_base.json")

SYSTEM_PROMPT = (
    "You are Lumen Travel's Cần Thơ assistant. "
    "Answer the user's question ONLY using the provided travel guide entries.\n"
    "Rules:\n"
    "1. Cite supporting entry IDs inline in square brackets, such as [G01] or [G03].\n"
    "2. Never invent opening hours, ticket prices, phone numbers, routes, or outside facts.\n"
    "3. If the provided entries do not contain the exact information needed to answer the question, "
    f'reply with exactly: "{IDK_RESPONSE}"'
)


class RAGError(Exception):
    """User-friendly error raised when the RAG pipeline cannot complete a request."""


def _normalize_base_url(base_url: str | None) -> str:
    """Validate and strip trailing slashes from the Ollama + Pinggy base URL."""
    resolved = (OLLAMA_BASE_URL if base_url is None else base_url).strip().rstrip("/")
    if not resolved:
        raise RAGError("OLLAMA_BASE_URL is not configured in .env.")
    if not resolved.startswith(("http://", "https://")):
        raise RAGError("OLLAMA_BASE_URL must start with http:// or https://.")
    return resolved


def _post_ollama(
    endpoint: str,
    payload: dict[str, Any],
    base_url: str | None = None,
    timeout: int = CHAT_TIMEOUT,
) -> dict[str, Any]:
    """Send one POST request to Ollama and map network or HTTP failures to RAGError."""
    url = f"{_normalize_base_url(base_url)}{endpoint}"
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException:
        raise RAGError(
            "Cannot reach the Ollama server. Check that your Colab + Pinggy tunnel is running "
            "and OLLAMA_BASE_URL in .env is up to date."
        ) from None

    try:
        data = response.json()
    except ValueError:
        if response.status_code in (502, 503, 504):
            raise RAGError(
                "The Ollama Pinggy tunnel returned a gateway error. Check that Ollama is running in Colab."
            ) from None
        raise RAGError(f"Ollama returned a non-JSON response (HTTP {response.status_code}).") from None

    if response.ok and isinstance(data, dict):
        return data

    error_detail = str(data.get("error", "")) if isinstance(data, dict) else ""
    if response.status_code == 404 or "not found" in error_detail.lower():
        model_name = payload.get("model", "the requested model")
        raise RAGError(
            f"Ollama could not find '{model_name}'. Run `!ollama pull {model_name}` in Colab."
        )
    raise RAGError(f"Ollama request failed (HTTP {response.status_code}): {error_detail or 'unknown error'}.")


def load_knowledge_base(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load travel guide entries from JSON."""
    kb_path = DEFAULT_KB_PATH if path is None else Path(path)
    if not kb_path.exists():
        raise RAGError(f"Knowledge base file not found: {kb_path.name}")
    try:
        entries = json.loads(kb_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RAGError(f"Could not read {kb_path.name}: {exc}") from None
    if not isinstance(entries, list) or not entries:
        raise RAGError(f"{kb_path.name} must contain a non-empty JSON array of entries.")
    return entries


def format_entry_for_embedding(entry: dict[str, Any]) -> str:
    """Combine structured metadata and body text into one string for embedding."""
    parts = [str(entry.get("title", "")).strip()]
    category = str(entry.get("category", "")).strip()
    if category:
        parts.append(f"Category: {category}")
    tags = entry.get("tags") or []
    if isinstance(tags, list) and tags:
        parts.append(f"Tags: {', '.join(str(tag).strip() for tag in tags if str(tag).strip())}")
    parts.append(str(entry.get("text", "")).strip())
    return " | ".join(part for part in parts if part)


def embed_texts(
    texts: list[str],
    base_url: str | None = None,
    model: str | None = None,
) -> np.ndarray:
    """Embed one or more strings using Ollama /api/embed (embeddinggemma by default)."""
    if not texts:
        raise RAGError("Cannot embed an empty list of texts.")
    resolved_model = (EMBED_MODEL if model is None else model).strip() or "embeddinggemma"
    data = _post_ollama(
        "/api/embed",
        {"model": resolved_model, "input": texts},
        base_url=base_url,
        timeout=EMBED_TIMEOUT,
    )
    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise RAGError("Ollama returned an unexpected embeddings payload.")
    return np.array(embeddings, dtype=float)


def retrieve_top_k(
    question: str,
    entries: list[dict[str, Any]],
    doc_vectors: np.ndarray,
    k: int = DEFAULT_TOP_K,
    base_url: str | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Embed the question and return top-k entries ranked by cosine similarity."""
    if not entries or len(entries) != len(doc_vectors):
        raise RAGError("Knowledge base entries do not match cached embedding vectors.")
    query_vector = embed_texts([question], base_url=base_url, model=model)[0]
    if query_vector.shape[0] != doc_vectors.shape[1]:
        raise RAGError("Query embedding dimension does not match document embeddings.")

    scores = (doc_vectors @ query_vector) / (
        np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector) + 1e-12
    )
    top_indices = np.argsort(scores)[::-1][: max(1, min(k, len(entries)))]
    return [
        {"entry": entries[int(idx)], "score": float(scores[int(idx)])}
        for idx in top_indices
    ]


def build_context(retrieved: list[dict[str, Any]]) -> str:
    """Format retrieved guide entries with IDs and metadata for the LLM prompt."""
    blocks = []
    for item in retrieved:
        entry = item["entry"]
        header = f"[{entry['id']}] {entry['title']}"
        if entry.get("category"):
            header += f" (category: {entry['category']})"
        blocks.append(
            f"{header}\n"
            f"Details: {entry['text']}\n"
            f"Source: {entry['source']}"
        )
    return "\n\n".join(blocks)


def _strip_thinking(raw_content: str) -> str:
    """Remove model reasoning blocks, including templates that omit the opening <think> tag."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", str(raw_content))
    # Some Ollama qwen3 templates prefill <think> in the prompt, leaving only a closing </think>.
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1]
    return re.sub(r"<think>[\s\S]*$", "", cleaned).strip()


def generate_answer(
    question: str,
    retrieved: list[dict[str, Any]],
    base_url: str | None = None,
    model: str | None = None,
) -> str:
    """Generate a grounded answer from retrieved entries using qwen3:4b."""
    resolved_model = (CHAT_MODEL if model is None else model).strip() or "qwen3:4b"
    context = build_context(retrieved)
    user_message = (
        f"Travel Guide Entries:\n{context}\n\n"
        f"Question: {question.strip()}\n\n"
        "Answer strictly from the travel guide entries above and cite [ID] tags."
    )
    data = _post_ollama(
        "/api/chat",
        {
            "model": resolved_model,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.1},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        },
        base_url=base_url,
        timeout=CHAT_TIMEOUT,
    )
    message = data.get("message")
    raw_content = message.get("content", "") if isinstance(message, dict) else ""
    return _strip_thinking(raw_content) or IDK_RESPONSE


def answer_question(
    question: str,
    entries: list[dict[str, Any]],
    doc_vectors: np.ndarray,
    k: int = DEFAULT_TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
    base_url: str | None = None,
    embed_model: str | None = None,
    chat_model: str | None = None,
) -> dict[str, Any]:
    """Run two-layer RAG: threshold guardrail at retrieval + grounded LLM generation."""
    clean_question = (question or "").strip()
    if not clean_question:
        raise RAGError("Please enter a non-empty question.")

    retrieved = retrieve_top_k(
        clean_question,
        entries,
        doc_vectors,
        k=k,
        base_url=base_url,
        model=embed_model,
    )
    top_score = retrieved[0]["score"] if retrieved else 0.0
    if top_score < threshold:
        return {
            "answer": IDK_RESPONSE,
            "retrieved": retrieved,
            "abstained_at_retrieval": True,
        }

    answer = generate_answer(
        clean_question,
        retrieved,
        base_url=base_url,
        model=chat_model,
    )
    return {
        "answer": answer,
        "retrieved": retrieved,
        "abstained_at_retrieval": False,
    }
