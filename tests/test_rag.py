from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from rag import (
    IDK_RESPONSE,
    RAGError,
    answer_question,
    build_context,
    format_entry_for_embedding,
    generate_answer,
    retrieve_top_k,
)

SAMPLE_ENTRIES = [
    {
        "id": "G01",
        "title": "Cai Rang Floating Market",
        "text": "Open 05:00 to 09:00.",
        "source": "Lumen",
        "category": "market",
        "tags": ["boat", "morning"],
    },
    {
        "id": "G03",
        "title": "Binh Thuy Ancient House",
        "text": "Ticket is 20,000 VND.",
        "source": "Lumen",
        "category": "heritage",
        "tags": ["history"],
    },
]
SAMPLE_VECTORS = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)


def test_format_entry_includes_category_and_tags():
    text = format_entry_for_embedding(SAMPLE_ENTRIES[0])
    assert "Cai Rang Floating Market" in text
    assert "market" in text
    assert "boat, morning" in text


def test_retrieve_top_k_ranks_by_cosine_similarity():
    with patch("rag.embed_texts", return_value=np.array([[0.1, 0.9]], dtype=float)):
        retrieved = retrieve_top_k(
            "How much is Binh Thuy ticket?",
            SAMPLE_ENTRIES,
            SAMPLE_VECTORS,
            k=2,
            base_url="https://example.pinggy.link",
        )
    assert [item["entry"]["id"] for item in retrieved] == ["G03", "G01"]
    assert retrieved[0]["score"] > retrieved[1]["score"]


def test_build_context_includes_ids_and_sources():
    context = build_context([{"entry": SAMPLE_ENTRIES[0], "score": 0.88}])
    assert "[G01]" in context
    assert "Cai Rang Floating Market" in context
    assert "Open 05:00 to 09:00." in context
    assert "Lumen" in context


def test_answer_question_rejects_empty_query():
    with pytest.raises(RAGError, match="Please enter a non-empty question"):
        answer_question("   ", SAMPLE_ENTRIES, SAMPLE_VECTORS)


def test_answer_question_abstains_at_retrieval_when_below_threshold():
    with patch("rag.embed_texts", return_value=np.array([[0.0, 0.0]], dtype=float)), \
         patch("rag.generate_answer") as mock_gen:
        result = answer_question(
            "What is the weather in Tokyo?",
            SAMPLE_ENTRIES,
            SAMPLE_VECTORS,
            k=2,
            threshold=0.35,
        )
        assert result["answer"] == IDK_RESPONSE
        assert result["abstained_at_retrieval"] is True
        assert len(result["retrieved"]) == 2
        mock_gen.assert_not_called()


def test_answer_question_calls_generation_when_above_threshold():
    with patch("rag.embed_texts", return_value=np.array([[0.95, 0.05]], dtype=float)), \
         patch("rag.generate_answer", return_value="It opens from 05:00 to 09:00 [G01].") as mock_gen:
        result = answer_question(
            "When does Cai Rang open?",
            SAMPLE_ENTRIES,
            SAMPLE_VECTORS,
            k=2,
            threshold=0.35,
        )
        assert result["answer"] == "It opens from 05:00 to 09:00 [G01]."
        assert result["abstained_at_retrieval"] is False
        mock_gen.assert_called_once()


def test_generate_answer_strips_think_tags_and_normalizes_url():
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {
        "message": {"content": "<think>internal reasoning</think>\nTickets cost 20,000 VND [G03]."}
    }
    retrieved = [{"entry": SAMPLE_ENTRIES[1], "score": 0.85}]
    with patch("rag.requests.post", return_value=mock_resp) as mock_post:
        ans = generate_answer(
            "How much is Binh Thuy ticket?",
            retrieved,
            base_url="https://example.pinggy.link/",
        )
        assert ans == "Tickets cost 20,000 VND [G03]."
        called_url = mock_post.call_args[0][0]
        assert called_url == "https://example.pinggy.link/api/chat"
        assert mock_post.call_args.kwargs["json"]["think"] is False


def test_generate_answer_strips_orphan_closing_think_tag():
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {
        "message": {"content": "Okay, let's check [G03].\n</think>\n\nTickets cost 20,000 VND [G03]."}
    }
    retrieved = [{"entry": SAMPLE_ENTRIES[1], "score": 0.85}]
    with patch("rag.requests.post", return_value=mock_resp):
        ans = generate_answer(
            "How much is Binh Thuy ticket?",
            retrieved,
            base_url="https://example.pinggy.link",
        )
        assert ans == "Tickets cost 20,000 VND [G03]."


def test_generate_answer_raises_rag_error_on_blank_url_or_connection_failure():
    retrieved = [{"entry": SAMPLE_ENTRIES[0], "score": 0.9}]
    with pytest.raises(RAGError, match="OLLAMA_BASE_URL"):
        generate_answer("Open hours?", retrieved, base_url="   ")

    with patch("rag.requests.post", side_effect=requests.ConnectionError("tunnel down")):
        with pytest.raises(RAGError, match="Cannot reach the Ollama server"):
            generate_answer("Open hours?", retrieved, base_url="https://example.pinggy.link")
