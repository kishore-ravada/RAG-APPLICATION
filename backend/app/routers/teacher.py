"""
Teacher Router — EduSaarthi AI
Provides teachers and admins with class-level dashboards, individual student
drill-down views, and teacher-in-the-loop intervention plan management.
"""
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.models import (
    User, UserRole, StudentProfile, QuizAttempt, LearningGap, LearningPlan,
    Topic, IndicatorStatus, PlanStatus, AuditLog
)
from app.schemas.schemas import (
    TeacherDashOut, StudentOverviewForTeacher, StudentDetailOut,
    PlanOut, PlanActionInput, GapOut, TrendPoint
)
from app.security.auth import get_current_user
from app.security.rbac import require_roles

router = APIRouter(prefix="/api/teacher", tags=["Teacher"])

_TEACHER_ROLES = [UserRole.TEACHER, UserRole.ADMIN]


@router.get("/dashboard", response_model=TeacherDashOut)
def get_teacher_dashboard(
    current_user: User = Depends(require_roles(_TEACHER_ROLES)),
    db: Session = Depends(get_db)
):
    """Returns a class-level overview with per-student performance summaries."""
    students = db.query(StudentProfile).all()
    total_students = len(students)

    needing_attention = 0
    on_track_count = 0
    all_scores = []
    student_overviews = []

    for sp in students:
        user = db.query(User).filter(User.id == sp.user_id).first()
        attempts = db.query(QuizAttempt).filter(QuizAttempt.student_id == sp.id).all()
        gaps = db.query(LearningGap).filter(LearningGap.student_id == sp.id).all()

        student_avg = (
            sum(a.score_percentage for a in attempts) / len(attempts)
            if attempts else 75.0
        )
        all_scores.append(student_avg)

        status_val = IndicatorStatus.ON_TRACK
        weakest_topic = "None"
        evidence_snip = "Demonstrating solid concept comprehension."
        recommended = "Maintain current curriculum pacing."

        for g in gaps:
            if g.indicator_status == IndicatorStatus.NEEDS_ATTENTION:
                status_val = IndicatorStatus.NEEDS_ATTENTION
                topic = db.query(Topic).filter(Topic.id == g.topic_id).first()
                weakest_topic = topic.name if topic else "SQL JOINs"
                evidence_snip = g.evidence_text or "Declining assessment scores."
                recommended = f"Assign 5-Day Targeted Practice on {weakest_topic}."
                break
            elif g.indicator_status == IndicatorStatus.STRONG:
                status_val = IndicatorStatus.STRONG

        if status_val == IndicatorStatus.NEEDS_ATTENTION:
            needing_attention += 1
        else:
            on_track_count += 1

        student_overviews.append(StudentOverviewForTeacher(
            student_id=sp.id,
            user_id=sp.user_id,
            full_name=user.full_name if user else f"Student #{sp.id}",
            grade_level=sp.grade_level,
            average_score=round(student_avg, 1),
            indicator_status=status_val,
            weakest_topic=weakest_topic,
            evidence_snippet=evidence_snip,
            recommended_action=recommended
        ))

    avg_class_score = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0.0

    return TeacherDashOut(
        total_students=total_students,
        students_on_track=on_track_count,
        students_needing_attention=needing_attention,
        average_class_score=avg_class_score,
        most_difficult_topics=["SQL JOIN Syntax", "Python Recursion"],
        students=student_overviews
    )


@router.get("/students/{student_id}", response_model=StudentDetailOut)
def get_student_detail(
    student_id: int,
    current_user: User = Depends(require_roles(_TEACHER_ROLES)),
    db: Session = Depends(get_db)
):
    """Returns a detailed performance breakdown for a single student."""
    sp = db.query(StudentProfile).filter(StudentProfile.id == student_id).first()
    if not sp:
        raise HTTPException(status_code=404, detail="Student not found")

    user = db.query(User).filter(User.id == sp.user_id).first()
    attempts = (
        db.query(QuizAttempt)
        .filter(QuizAttempt.student_id == sp.id)
        .order_by(QuizAttempt.completed_at.asc())
        .all()
    )
    gaps = db.query(LearningGap).filter(LearningGap.student_id == sp.id).all()
    plans = db.query(LearningPlan).filter(LearningPlan.student_id == sp.id).all()

    avg_score = (
        round(sum(a.score_percentage for a in attempts) / len(attempts), 1)
        if attempts else 0.0
    )

    score_history = [
        TrendPoint(
            attempt_number=idx + 1,
            score=a.score_percentage,
            date=a.completed_at.strftime("%b %d")
        )
        for idx, a in enumerate(attempts)
    ]

    gap_list = []
    status_val = IndicatorStatus.ON_TRACK
    for g in gaps:
        topic = db.query(Topic).filter(Topic.id == g.topic_id).first()
        if g.indicator_status == IndicatorStatus.NEEDS_ATTENTION:
            status_val = IndicatorStatus.NEEDS_ATTENTION
        gap_list.append(GapOut(
            id=g.id,
            topic_name=topic.name if topic else "Topic",
            indicator_status=g.indicator_status,
            accuracy_percentage=g.accuracy_percentage,
            repeated_mistakes_count=g.repeated_mistakes_count,
            evidence_text=g.evidence_text or "Recorded performance trend",
            updated_at=g.updated_at
        ))

    plan_list = []
    for p in plans:
        topic = db.query(Topic).filter(Topic.id == p.topic_id).first()
        plan_list.append(PlanOut(
            id=p.id,
            student_id=p.student_id,
            topic_id=p.topic_id,
            topic_name=topic.name if topic else "DBMS & SQL JOINs",
            title=p.title,
            status=p.status,
            created_by=p.created_by,
            teacher_feedback=p.teacher_feedback,
            created_at=p.created_at,
            items=p.items
        ))

    return StudentDetailOut(
        student_id=sp.id,
        full_name=user.full_name if user else "Student",
        grade_level=sp.grade_level,
        indicator_status=status_val,
        average_score=avg_score,
        score_history=score_history,
        learning_gaps=gap_list,
        active_plans=plan_list
    )


@router.patch("/interventions/{plan_id}", response_model=PlanOut)
def update_teacher_intervention(
    plan_id: int,
    action_in: PlanActionInput,
    request: Request,
    current_user: User = Depends(require_roles(_TEACHER_ROLES)),
    db: Session = Depends(get_db)
):
    """
    Teacher-in-the-loop: approve, edit, or reject an AI-generated learning plan.
    All intervention decisions are recorded in the audit trail.
    """
    plan = db.query(LearningPlan).filter(LearningPlan.id == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Learning plan not found")

    act = action_in.action.upper()
    if act == "APPROVE":
        plan.status = PlanStatus.APPROVED
    elif act == "REJECT":
        plan.status = PlanStatus.REJECTED
    elif act == "EDIT":
        plan.status = PlanStatus.APPROVED
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid action. Use APPROVE, REJECT, or EDIT."
        )

    if action_in.teacher_feedback:
        plan.teacher_feedback = action_in.teacher_feedback

    audit = AuditLog(
        user_id=current_user.id,
        user_email=current_user.email,
        action=f"TEACHER_INTERVENTION_{act}",
        details=f"Teacher {current_user.full_name} set plan #{plan_id} to {plan.status.value}",
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(audit)
    db.commit()
    db.refresh(plan)

    topic = db.query(Topic).filter(Topic.id == plan.topic_id).first()
    plan.topic_name = topic.name if topic else "Topic"
    return plan
