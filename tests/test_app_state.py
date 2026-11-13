from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import streamlit as st
from streamlit.testing.v1 import AppTest

from app import forget_user

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_forget_user_clears_user_and_chat_history():
    fake_state = {
        "user": {"uid": "u1", "email": "a@b.com"},
        "chat_history": [{"role": "user", "content": "hi"}],
        "auth_password": "secret",
        "auth_email": "a@b.com",
        "unrelated_key": "keep_me",
    }
    forget_user(fake_state)
    assert "user" not in fake_state
    assert "chat_history" not in fake_state
    assert "auth_password" not in fake_state
    assert "auth_email" not in fake_state
    assert fake_state["unrelated_key"] == "keep_me"


def test_chat_page_is_hidden_until_sign_in():
    with patch("firebase_client.FIREBASE_PROJECT_ID", "demo-proj"), \
         patch("firebase_client.FIREBASE_API_KEY", "fake-key"):
        at = AppTest.from_file(APP_PATH).run()
        assert not at.exception
        button_labels = [btn.label for btn in at.button]
        assert "Sign in" in button_labels
        assert "Create account" in button_labels
        assert "Sign out" not in button_labels
        assert "Ask" not in button_labels


def test_signed_in_flow_warns_on_empty_question_renders_answer_and_clears_on_sign_out():
    fake_vectors = np.ones((16, 4), dtype=float)
    fake_query_vec = np.ones((1, 4), dtype=float)
    st.cache_resource.clear()
    try:
        with patch("firebase_client.FIREBASE_PROJECT_ID", "demo-proj"), \
             patch("firebase_client.FIREBASE_API_KEY", "fake-key"), \
             patch("rag.embed_texts", side_effect=[fake_vectors, fake_query_vec]), \
             patch("rag.generate_answer", return_value="Cái Răng opens from 05:00 to 09:00 [G01]."):
            at = AppTest.from_file(APP_PATH)
            at.session_state["user"] = {
                "uid": "u1",
                "email": "student@hcmus.edu.vn",
                "id_token": "tok",
                "expires_at": time.time() + 3600,
            }
            at.session_state["chat_history"] = []
            at.run()
            assert not at.exception
            assert any(btn.label == "Sign out" for btn in at.button)

            # Submit blank question -> warning shown, chat_history unchanged
            at.text_input[0].input("   ")
            ask_btn = next(btn for btn in at.button if btn.label == "Ask")
            ask_btn.click().run()
            assert any("non-empty question" in w.value for w in at.warning)
            assert len(at.session_state["chat_history"]) == 0

            # Submit valid question -> answer + expander rendered in chat_history
            at.text_input[0].input("When does Cai Rang open?")
            ask_btn = next(btn for btn in at.button if btn.label == "Ask")
            ask_btn.click().run()
            assert not at.exception
            assert len(at.session_state["chat_history"]) == 2
            assert any(exp.label == "Retrieved Guide Entries & Scores" for exp in at.expander)

            # Sign out -> clears user and chat_history and returns to sign-in gate
            sign_out_btn = next(btn for btn in at.button if btn.label == "Sign out")
            sign_out_btn.click().run()
            assert "user" not in at.session_state
            assert "chat_history" not in at.session_state
            assert any(btn.label == "Sign in" for btn in at.button)
    finally:
        st.cache_resource.clear()
