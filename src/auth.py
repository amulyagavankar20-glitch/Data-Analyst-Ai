"""PostgreSQL-backed email/password authentication."""
import hashlib
import logging
import os
from pathlib import Path
import re
import secrets
import time

import psycopg
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
PASSWORD_HASHER = PasswordHasher()
SESSION_TTL_SECONDS = 12 * 60 * 60
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class AuthUnavailable(RuntimeError):
    """Authentication storage could not be reached or initialized."""


def _database_url():
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise AuthUnavailable("DATABASE_URL is not configured.")
    return url


def _connect():
    try:
        return psycopg.connect(_database_url(), connect_timeout=5,
                               application_name="ai-data-analyst-auth")
    except AuthUnavailable:
        raise
    except psycopg.Error as exc:
        logger.warning("PostgreSQL connection failed: %s", type(exc).__name__)
        raise AuthUnavailable("Authentication storage is unavailable.") from exc


def initialize_auth_db():
    """Apply the checked-in SQL schema. The application DB role needs DDL rights."""
    schema_path = Path(__file__).resolve().parent.parent / "schema.sql"
    try:
        statements = [statement.strip() for statement in schema_path.read_text(encoding="utf-8").split(";")
                      if statement.strip()]
    except OSError as exc:
        logger.exception("PostgreSQL schema file could not be read")
        raise AuthUnavailable("Authentication schema is unavailable.") from exc
    try:
        with _connect() as connection:
            for statement in statements:
                connection.execute(statement)
    except AuthUnavailable:
        raise
    except psycopg.Error as exc:
        logger.warning("PostgreSQL auth schema initialization failed: %s", type(exc).__name__)
        raise AuthUnavailable("Authentication storage could not be initialized.") from exc


def _clean_email(email):
    normalized = (email or "").strip().lower()
    if len(normalized) > 254 or not EMAIL_PATTERN.fullmatch(normalized):
        raise ValueError("Enter a valid email address.")
    return normalized


def _validate_password(password):
    if not isinstance(password, str) or len(password) < 12:
        raise ValueError("Use a password with at least 12 characters.")
    if len(password) > 128:
        raise ValueError("Password must be 128 characters or fewer.")


def _token_hash(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _create_session(connection, user_id, email):
    connection.execute("DELETE FROM auth_sessions WHERE expires_at < NOW()")
    token = secrets.token_urlsafe(32)
    token_digest = _token_hash(token)
    expires_at = int(time.time()) + SESSION_TTL_SECONDS
    connection.execute(
        "INSERT INTO auth_sessions (token_hash, user_id, expires_at) "
        "VALUES (%s, %s, to_timestamp(%s))",
        (token_digest, user_id, expires_at),
    )
    return {"user_id": user_id, "email": email, "session_token": token,
            "expires_at": expires_at}


def sign_up_with_email_and_password(email, password):
    """Create an account and issue an opaque server-validated session."""
    try:
        normalized_email = _clean_email(email)
        _validate_password(password)
    except ValueError as exc:
        return {"error": {"message": str(exc)}}
    password_hash = PASSWORD_HASHER.hash(password)
    try:
        with _connect() as connection:
            row = connection.execute(
                "INSERT INTO auth_users (email, password_hash) VALUES (%s, %s) "
                "ON CONFLICT (email) DO NOTHING RETURNING id, email",
                (normalized_email, password_hash),
            ).fetchone()
            if row is None:
                return {"error": {"message": "An account with that email already exists."}}
            return _create_session(connection, row[0], row[1])
    except AuthUnavailable:
        return {"error": {"message": "Authentication storage is unavailable. Please try again."}}
    except psycopg.Error as exc:
        logger.warning("PostgreSQL signup failed: %s", type(exc).__name__)
        return {"error": {"message": "Account creation failed. Please try again."}}


def sign_in_with_email_and_password(email, password):
    """Verify a password and issue an opaque server-validated session."""
    try:
        normalized_email = _clean_email(email)
        _validate_password(password)
    except ValueError:
        return {"error": {"message": "Email or password is incorrect."}}
    try:
        with _connect() as connection:
            row = connection.execute(
                "SELECT id, email, password_hash FROM auth_users WHERE email = %s",
                (normalized_email,),
            ).fetchone()
            if row is None:
                return {"error": {"message": "Email or password is incorrect."}}
            try:
                PASSWORD_HASHER.verify(row[2], password)
            except (VerifyMismatchError, VerificationError, InvalidHashError):
                return {"error": {"message": "Email or password is incorrect."}}
            if PASSWORD_HASHER.check_needs_rehash(row[2]):
                connection.execute("UPDATE auth_users SET password_hash = %s WHERE id = %s",
                                   (PASSWORD_HASHER.hash(password), row[0]))
            return _create_session(connection, row[0], row[1])
    except AuthUnavailable:
        return {"error": {"message": "Authentication storage is unavailable. Please try again."}}
    except psycopg.Error as exc:
        logger.warning("PostgreSQL sign-in failed: %s", type(exc).__name__)
        return {"error": {"message": "Sign-in failed. Please try again."}}


def validate_session(session):
    """Return the account only while its opaque session token remains valid."""
    token = (session or {}).get("session_token")
    if not token:
        return None
    try:
        with _connect() as connection:
            row = connection.execute(
                "SELECT u.id, u.email FROM auth_sessions s "
                "JOIN auth_users u ON u.id = s.user_id "
                "WHERE s.token_hash = %s AND s.expires_at > NOW()",
                (_token_hash(token),),
            ).fetchone()
            if row is None:
                return None
            return {**session, "user_id": row[0], "email": row[1]}
    except AuthUnavailable:
        raise
    except psycopg.Error as exc:
        logger.warning("PostgreSQL session validation failed: %s", type(exc).__name__)
        raise AuthUnavailable("Authentication storage is unavailable.") from exc


def revoke_session(session):
    """Revoke a session on logout. Safe to call with an empty session."""
    token = (session or {}).get("session_token")
    if not token:
        return
    try:
        with _connect() as connection:
            connection.execute("DELETE FROM auth_sessions WHERE token_hash = %s", (_token_hash(token),))
    except AuthUnavailable:
        raise
    except psycopg.Error as exc:
        logger.warning("PostgreSQL logout failed: %s", type(exc).__name__)
        raise AuthUnavailable("Authentication storage is unavailable.") from exc
