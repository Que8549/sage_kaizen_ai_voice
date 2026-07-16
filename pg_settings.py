from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to this file, not the process CWD.
# pg_settings.py lives at the project root, so this always points to
# <project_root>/.env regardless of where the script is invoked from.
#
# 2026-07-16: added as this project's first-ever Postgres touchpoint, to
# support PostgresLogHandler in sk_logging.py (see log/db/log_schema.sql in
# the main project). Mirrors sage_kaizen_ai / sage_kaizen_ai_ingest's
# pg_settings.py exactly — same "local copy per project" convention already
# used for sk_logging.py.
_ENV_FILE = Path(__file__).resolve().parent / ".env"


class PgSettings(BaseSettings):
    """Shared PostgreSQL connection fields.

    Values are populated in this order:
        1. .env file (project root — resolved relative to this file)
        2. OS environment variables
        3. Default values defined below

    Example .env file (project root):

        PG_USER=sage
        PG_PASSWORD=YourRealPassword
        PG_DB=sage_kaizen
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    pg_user: str = "my_user"
    pg_password: str = "my_pwd"
    pg_host: str = "127.0.0.1"
    pg_port: int = 5432
    pg_db: str = "my_db"

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )
