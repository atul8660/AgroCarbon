import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

# Determine dialect (SQLite for rapid-dev, Postgres for cloud)
# This will safely default to sqlite locally while you test
if hasattr(settings, "DATABASE_URI") and settings.DATABASE_URI:
    SQLALCHEMY_DATABASE_URL = settings.DATABASE_URI
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    SQLALCHEMY_DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'sql_app.db')}"

# SQLite requires specific arguments for FastAPI multi-threading
connect_args = {"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {}

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args=connect_args
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
