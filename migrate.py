"""Apply versioned database migrations as an operator, not at app startup."""
from pathlib import Path
import os
import psycopg


url = os.getenv("MIGRATION_DATABASE_URL")
if not url:
    raise SystemExit("Set MIGRATION_DATABASE_URL to a temporary schema-owner connection.")
sql = Path(__file__).parent / "migrations" / "001_auth.sql"
with psycopg.connect(url, connect_timeout=5, application_name="ai-data-analyst-migration") as connection:
    connection.execute(sql.read_text(encoding="utf-8"))
print("Applied migrations/001_auth.sql")
