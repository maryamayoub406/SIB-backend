"""
Database connection and session management.
"""
import os
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "sib_battery")
DATABASE_URL = os.getenv("DATABASE_URL")
SQLITE_PATH = os.getenv("SQLITE_PATH", str(BASE_DIR / "sib_battery.db"))


def _sqlite_url() -> str:
    return f"sqlite:///{Path(SQLITE_PATH).resolve().as_posix()}"


def _mysql_url() -> str:
    return (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        "?charset=utf8mb4"
    )


def _create_sqlite_engine():
    return create_engine(
        _sqlite_url(),
        connect_args={"check_same_thread": False},
        echo=False,
    )


def _create_engine():
    if DATABASE_URL and DATABASE_URL.startswith("sqlite"):
        print(f"[DB] Using configured SQLite database: {DATABASE_URL}")
        return create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},
            echo=False,
        )

    preferred_url = DATABASE_URL or _mysql_url()
    try:
        engine = create_engine(
            preferred_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_timeout=5,
            connect_args={"connect_timeout": 5},
            echo=False,
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"[DB] Using MySQL database at {DB_HOST}:{DB_PORT}/{DB_NAME}")
        return engine
    except Exception as exc:
        sqlite_engine = _create_sqlite_engine()
        print(f"[DB] MySQL unavailable, falling back to SQLite at {_sqlite_url()}: {exc}")
        return sqlite_engine


engine = _create_engine()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> bool:
    """Check if database is reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"[DB] Connection failed: {e}")
        return False
