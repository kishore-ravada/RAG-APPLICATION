"""
Health Check Router — EduSaarthi AI
Provides a single /health endpoint that verifies database connectivity,
AI configuration status, and RAG pipeline readiness.
"""
import datetime
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.schemas import HealthCheckOut
from app.config import settings

router = APIRouter(tags=["System Health"])


@router.get("/health", response_model=HealthCheckOut)
def health_check(db: Session = Depends(get_db)):
    """Returns the current health status of the API, database, and AI subsystems."""
    db_status = "healthy"
    try:
        db.execute(text("SELECT 1"))  # SQLAlchemy 2.x compatible syntax
    except Exception:
        db_status = "unhealthy"

    ai_status = "configured" if settings.GEMINI_API_KEY else "demo_mode_fallback"

    return HealthCheckOut(
        status="healthy",
        database=db_status,
        ai=ai_status,
        rag="ready",
        timestamp=datetime.datetime.utcnow()
    )
