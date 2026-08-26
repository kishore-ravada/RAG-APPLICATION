"""
AI Tutor Router — EduSaarthi AI
Exposes the RAG-grounded AI tutoring endpoint with prompt injection
protection, rate limiting, and full audit logging.
"""
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.schemas import TutorRequest, TutorResponse
from app.models.models import User, AuditLog
from app.security.auth import get_current_user
from app.security.rate_limiter import rate_limit
from app.rag.rag_pipeline import query_rag_pipeline

router = APIRouter(prefix="/api/tutor", tags=["AI Tutor"])


@router.post("/ask", response_model=TutorResponse)
def ask_tutor(
    tutor_in: TutorRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Submits a question to the RAG-grounded AI tutor.
    Input is sanitised and checked for prompt injection before querying.
    Rate limited to 20 requests per minute per IP.
    """
    rate_limit(request, max_requests=20, window_seconds=60)

    client_ip = request.client.host if request.client else "127.0.0.1"
    response = query_rag_pipeline(tutor_in.question, tutor_in.level, client_ip=client_ip, db=db)

    audit = AuditLog(
        user_id=current_user.id,
        user_email=current_user.email,
        action="AI_TUTOR_QUERY",
        details=f"Question: '{tutor_in.question[:60]}...' | Level: {tutor_in.level}",
        ip_address=client_ip
    )
    db.add(audit)
    db.commit()

    return TutorResponse(**response)
