from unittest.mock import MagicMock, patch

import pytest
import requests

from firebase_client import FirebaseError, create_account, sign_in, validate_config


def test_validate_config_rejects_empty_api_key():
    with pytest.raises(FirebaseError, match="Copy .env.example to .env"):
        validate_config(api_key="", project_id="demo-project")


def test_validate_config_rejects_invalid_project_id():
    with pytest.raises(FirebaseError, match="Project ID should be the ID"):
        validate_config(api_key="fake-key", project_id="https://bad/project")


def test_sign_in_rejects_empty_credentials():
    with patch("firebase_client.FIREBASE_API_KEY", "fake-key"), \
         patch("firebase_client.FIREBASE_PROJECT_ID", "demo-proj"):
        with pytest.raises(FirebaseError, match="Enter both email and password"):
            sign_in("   ", "secret123")


def test_sign_in_returns_session_dict():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.ok = True
    mock_resp.json.return_value = {
        "localId": "uid-123",
        "email": "student@hcmus.edu.vn",
        "idToken": "token-abc",
        "expiresIn": "3600",
    }
    with patch("firebase_client.FIREBASE_API_KEY", "fake-key"), \
         patch("firebase_client.FIREBASE_PROJECT_ID", "demo-proj"), \
         patch("firebase_client.requests.request", return_value=mock_resp):
        session = sign_in("student@hcmus.edu.vn", "password123")
        assert session["uid"] == "uid-123"
        assert session["email"] == "student@hcmus.edu.vn"
        assert session["id_token"] == "token-abc"
        assert session["expires_at"] > 0


def test_sign_in_maps_invalid_credentials_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.ok = False
    mock_resp.json.return_value = {"error": {"message": "INVALID_LOGIN_CREDENTIALS"}}
    with patch("firebase_client.FIREBASE_API_KEY", "fake-key"), \
         patch("firebase_client.FIREBASE_PROJECT_ID", "demo-proj"), \
         patch("firebase_client.requests.request", return_value=mock_resp):
        with pytest.raises(FirebaseError, match="Sign-in failed"):
            sign_in("student@hcmus.edu.vn", "wrongpass")


def test_create_account_maps_existing_email_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.ok = False
    mock_resp.json.return_value = {"error": {"message": "EMAIL_EXISTS"}}
    with patch("firebase_client.FIREBASE_API_KEY", "fake-key"), \
         patch("firebase_client.FIREBASE_PROJECT_ID", "demo-proj"), \
         patch("firebase_client.requests.request", return_value=mock_resp):
        with pytest.raises(FirebaseError, match="already registered"):
            create_account("student@hcmus.edu.vn", "password123")


def test_sign_in_handles_network_exception_without_leaking_key():
    with patch("firebase_client.FIREBASE_API_KEY", "super-secret-key"), \
         patch("firebase_client.FIREBASE_PROJECT_ID", "demo-proj"), \
         patch(
             "firebase_client.requests.request",
             side_effect=requests.ConnectionError("https://identitytoolkit.googleapis.com?key=super-secret-key"),
         ):
        with pytest.raises(FirebaseError, match="Cannot reach Firebase") as exc_info:
            sign_in("student@hcmus.edu.vn", "password123")
        assert "super-secret-key" not in str(exc_info.value)
