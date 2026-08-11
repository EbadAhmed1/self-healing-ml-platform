"""
app/database.py
───────────────
SQLAlchemy engine + session factory.

SessionLocal is used as a FastAPI dependency (via get_db()) so that every
request gets its own database session that is properly closed afterward.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings


def _make_engine():
    settings = get_settings()
    url = str(settings.database_url)

    # psycopg2 requires the scheme to be "postgresql" not "postgres"
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    connect_args = {}
    kwargs = {"pool_pre_ping": True}

    # SQLite (used in tests) needs check_same_thread=False and doesn't use pool_size
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    else:
        # Sensible pooling settings for free-tier Postgres (Supabase connection limits)
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 10
        kwargs["pool_recycle"] = 1800

    return create_engine(url, connect_args=connect_args, **kwargs)


engine = _make_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------
def get_db():
    """Yield a database session, ensuring it is closed after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
