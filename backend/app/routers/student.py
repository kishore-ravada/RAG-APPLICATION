"""
Student Router — EduSaarthi AI
Endpoints for student dashboard, learning gap retrieval, and personalised
intervention plan generation.
"""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.models import (
    User, UserRole, StudentProfile, QuizAttempt,
    LearningGap, LearningPlan, Topic, IndicatorStatus
)
from app.schemas.schemas import StudentDashOut, GapOut, PlanOut, TrendPoint
from app.security.auth import get_current_user
from app.security.rbac import require_roles
from app.services.plan_service import create_or_get_learning_plan

router = APIRouter(prefix="/api/student", tags=["Student"])

# Roles allowed to view student data (students see own, teachers/admins can preview)
_STUDENT_ROLES = [UserRole.STUDENT, UserRole.TEACHER, UserRole.ADMIN]

# Default topic for demo/preview purposes (DBMS & SQL JOINs)
_DEFAULT_TOPIC_ID = 2


@router.get("/dashboard", response_model=StudentDashOut)
def get_student_dashboard(
    current_user: User = Depends(require_roles(_STUDENT_ROLES)),
    db: Session = Depends(get_db)
):
    """Returns the authenticated student's dashboard summary."""
    sp = current_user.student_profile
    if not sp:
        # Allow teachers/admins to preview the first available student profile
        sp = db.query(StudentProfile).first()
        if not sp:
            raise HTTPException(status_code=404, detail="Student profile not found")

    attempts = (
        db.query(QuizAttempt)
        .filter(QuizAttempt.student_id == sp.id)
        .order_by(QuizAttempt.completed_at.asc())
        .all()
    )
    gaps = db.query(LearningGap).filter(LearningGap.student_id == sp.id).all()

    avg_score = (
        round(sum(a.score_percentage for a in attempts) / len(attempts), 1)
        if attempts else 0.0
    )

    trend = [
        TrendPoint(
            attempt_number=idx + 1,
            score=a.score_percentage,
            date=a.completed_at.strftime("%b %d")
        )
        for idx, a in enumerate(attempts)
    ]

    strong_topics: List[str] = []
    weak_topics: List[str] = []
    current_status = IndicatorStatus.ON_TRACK

    for g in gaps:
        topic = db.query(Topic).filter(Topic.id == g.topic_id).first()
        tname = topic.name if topic else "Topic"
        if g.indicator_status == IndicatorStatus.NEEDS_ATTENTION:
            weak_topics.append(tname)
            current_status = IndicatorStatus.NEEDS_ATTENTION
        else:
            strong_topics.append(tname)

    active_plan = (
        db.query(LearningPlan)
        .filter(LearningPlan.student_id == sp.id)
        .first()
    )
    if active_plan:
        topic = db.query(Topic).filter(Topic.id == active_plan.topic_id).first()
        active_plan.topic_name = topic.name if topic else "SQL JOIN Operations"

    return StudentDashOut(
        student_name=current_user.full_name,
        grade_level=sp.grade_level,
        indicator_status=current_status,
        average_score=avg_score,
        strong_topics=strong_topics or ["Python Variables", "DBMS Syntax"],
        weak_topics=weak_topics or ["SQL JOINs"],
        recent_trend=trend,
        active_plan=active_plan
    )


@router.get("/learning-gaps", response_model=List[GapOut])
def get_student_learning_gaps(
    current_user: User = Depends(require_roles(_STUDENT_ROLES)),
    db: Session = Depends(get_db)
):
    """Returns all detected learning gaps for the current student."""
    sp = current_user.student_profile
    if not sp:
        return []

    gaps = db.query(LearningGap).filter(LearningGap.student_id == sp.id).all()
    results = []
    for g in gaps:
        topic = db.query(Topic).filter(Topic.id == g.topic_id).first()
        results.append(GapOut(
            id=g.id,
            topic_name=topic.name if topic else "Unknown Topic",
            indicator_status=g.indicator_status,
            accuracy_percentage=g.accuracy_percentage,
            repeated_mistakes_count=g.repeated_mistakes_count,
            evidence_text=g.evidence_text or "No specific evidence recorded.",
            updated_at=g.updated_at
        ))
    return results


@router.post("/learning-plan", response_model=PlanOut)
def generate_student_learning_plan(
    topic_id: Optional[int] = _DEFAULT_TOPIC_ID,
    current_user: User = Depends(require_roles(_STUDENT_ROLES)),
    db: Session = Depends(get_db)
):
    """Generates or retrieves a 5-day AI-powered learning intervention plan."""
    sp = current_user.student_profile
    if not sp:
        # Fallback: use first available profile for teacher/admin preview
        sp = db.query(StudentProfile).first()

    plan = create_or_get_learning_plan(db, sp.id, topic_id)
    topic = db.query(Topic).filter(Topic.id == plan.topic_id).first()
    plan.topic_name = topic.name if topic else "DBMS & SQL JOINs"
    return plan
