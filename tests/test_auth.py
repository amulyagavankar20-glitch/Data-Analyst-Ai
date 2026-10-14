from src import auth


class Cursor:
    def __init__(self, row=None):
        self.row = row

    def fetchone(self):
        return self.row


class FakeConnection:
    def __init__(self, shared):
        self.shared = shared

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params=()):
        if query.startswith("INSERT INTO auth_users"):
            email, password_hash = params
            if email in self.shared["users_by_email"]:
                return Cursor()
            user_id = len(self.shared["users_by_email"]) + 1
            self.shared["users_by_email"][email] = (user_id, email, password_hash)
            return Cursor((user_id, email))
        if query.startswith("SELECT id, email, password_hash"):
            return Cursor(self.shared["users_by_email"].get(params[0]))
        if query.startswith("INSERT INTO auth_sessions"):
            self.shared["sessions"].add(params[0])
            return Cursor()
        if query.startswith("SELECT u.id, u.email FROM auth_sessions"):
            token_hash = params[0]
            user = next((row for row in self.shared["users_by_email"].values()
                         if row[0] and token_hash in self.shared["sessions"]), None)
            return Cursor((user[0], user[1]) if user else None)
        if query.startswith("DELETE FROM auth_sessions"):
            if params:
                self.shared["sessions"].discard(params[0])
            return Cursor()
        return Cursor()


def setup_fake_db(monkeypatch):
    shared = {"users_by_email": {}, "sessions": set()}
    monkeypatch.setattr(auth, "_connect", lambda: FakeConnection(shared))
    return shared


def test_signup_hashes_password_and_issues_valid_session(monkeypatch):
    shared = setup_fake_db(monkeypatch)
    created = auth.sign_up_with_email_and_password(" User@Example.com ", "a-long-password-123")
    assert created["email"] == "user@example.com"
    stored = shared["users_by_email"]["user@example.com"]
    assert stored[2] != "a-long-password-123"
    assert auth.PASSWORD_HASHER.verify(stored[2], "a-long-password-123")
    validated = auth.validate_session(created)
    assert validated["user_id"] == created["user_id"]


def test_password_policy_and_duplicate_registration(monkeypatch):
    setup_fake_db(monkeypatch)
    short = auth.sign_up_with_email_and_password("user@example.com", "short")
    assert "error" in short
    auth.sign_up_with_email_and_password("user@example.com", "a-long-password-123")
    duplicate = auth.sign_up_with_email_and_password("USER@example.com", "another-long-pass")
    assert "already exists" in duplicate["error"]["message"]


def test_login_rejects_wrong_password_and_logout_revokes_session(monkeypatch):
    shared = setup_fake_db(monkeypatch)
    auth.sign_up_with_email_and_password("user@example.com", "a-long-password-123")
    invalid = auth.sign_in_with_email_and_password("user@example.com", "wrong-password")
    assert "error" in invalid
    login = auth.sign_in_with_email_and_password("USER@example.com", "a-long-password-123")
    assert login.get("session_token")
    auth.revoke_session(login)
    assert auth.validate_session(login) is None
