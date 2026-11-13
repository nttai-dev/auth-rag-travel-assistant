"""Firebase Authentication REST client adapted from Lab 3 (no Firestore)."""

import time
from typing import Any

import requests

from config import FIREBASE_API_KEY, FIREBASE_PROJECT_ID

REQUEST_TIMEOUT = 15

_AUTH_ERROR_MESSAGES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("EMAIL_EXISTS",), "That email is already registered. Choose Sign in instead."),
    (
        ("OPERATION_NOT_ALLOWED",),
        "Enable Email/Password in Firebase Console > Authentication > Sign-in method.",
    ),
    (
        ("WEAK_PASSWORD", "PASSWORD_DOES_NOT_MEET_REQUIREMENTS"),
        "Choose a password that meets this Firebase project's password policy.",
    ),
    (("INVALID_EMAIL",), "Enter a valid email address."),
    (
        ("INVALID_LOGIN_CREDENTIALS", "INVALID_PASSWORD", "EMAIL_NOT_FOUND"),
        "Sign-in failed. Check your app email and password.",
    ),
    (("API_KEY", "PROJECT_NOT_FOUND"), "Check Project ID and Web API Key in .env."),
)


class FirebaseError(Exception):
    """User-facing Firebase error that avoids exposing raw API keys or payloads."""

    def __init__(self, message: str, sign_in_again: bool = False) -> None:
        super().__init__(message)
        self.sign_in_again = sign_in_again


def validate_config(api_key: str | None = None, project_id: str | None = None) -> None:
    """Validate local Firebase configuration without making network requests."""
    resolved_key = (FIREBASE_API_KEY if api_key is None else api_key).strip()
    resolved_project = (FIREBASE_PROJECT_ID if project_id is None else project_id).strip()
    if not resolved_project or not resolved_key:
        raise FirebaseError(
            "Copy .env.example to .env, fill in Project ID and Web API Key, then restart the app."
        )
    if any(ch in resolved_project for ch in "/ ?#"):
        raise FirebaseError(
            "Project ID should be the ID from Firebase Console, not a URL or project name."
        )


def _map_auth_error(code: str, status_code: int) -> FirebaseError:
    """Translate Firebase Identity Toolkit error codes into clear UI messages."""
    if any(marker in code for marker in ("TOKEN_EXPIRED", "INVALID_ID_TOKEN", "USER_DISABLED")) or status_code == 401:
        return FirebaseError("Your session is no longer valid. Please sign in again.", sign_in_again=True)
    if "TOO_MANY_ATTEMPTS" in code or status_code == 429:
        return FirebaseError("Too many requests. Wait a little before trying again.")
    for markers, message in _AUTH_ERROR_MESSAGES:
        if any(marker in code for marker in markers):
            return FirebaseError(message)
    if status_code == 403:
        return FirebaseError("Access denied. Check the Firebase project and API settings.")
    if status_code == 404:
        return FirebaseError("Check Project ID and Web API Key in .env.")
    return FirebaseError("Firebase could not complete the request. Check the setup in README.md.")


def _request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    """Send an HTTP request without propagating exceptions that include the API key URL."""
    try:
        response = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
    except requests.RequestException:
        raise FirebaseError("Cannot reach Firebase. Check your connection and try again.") from None

    try:
        data = response.json()
    except ValueError:
        raise FirebaseError("Firebase returned an unexpected response. Try again later.") from None
    if not isinstance(data, dict):
        raise FirebaseError("Firebase returned an unexpected response. Try again later.")
    if response.ok:
        return data

    error = data.get("error", {})
    code = str(error.get("message", "")) if isinstance(error, dict) else ""
    raise _map_auth_error(code, response.status_code)


def _authenticate(action: str, email: str, password: str) -> dict[str, Any]:
    """Call Firebase Identity Toolkit signUp or signInWithPassword endpoint."""
    validate_config()
    clean_email = email.strip()
    if not clean_email or not password:
        raise FirebaseError("Enter both email and password.")
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:{action}"
    return _request(
        "POST",
        url,
        params={"key": FIREBASE_API_KEY.strip()},
        json={"email": clean_email, "password": password, "returnSecureToken": True},
    )


def create_account(email: str, password: str) -> None:
    """Register an app user with Email/Password; users sign in separately afterward."""
    _authenticate("signUp", email, password)


def _session_from_response(data: dict[str, Any]) -> dict[str, Any]:
    """Extract a minimal session dictionary from Firebase sign-in response."""
    try:
        user = {
            "uid": data["localId"],
            "email": data["email"],
            "id_token": data["idToken"],
            "expires_at": time.time() + int(data["expiresIn"]),
        }
        if not all(isinstance(user[k], str) and user[k] for k in ("uid", "email", "id_token")):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise FirebaseError("Sign-in returned incomplete data. Please try again.") from None
    return user


def sign_in(email: str, password: str) -> dict[str, Any]:
    """Authenticate an app user and return session metadata (never the password)."""
    data = _authenticate("signInWithPassword", email, password)
    return _session_from_response(data)
