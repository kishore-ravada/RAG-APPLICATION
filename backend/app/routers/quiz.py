"""
Quiz Router — EduSaarthi AI
Endpoints for quiz generation and submission with automated learning-gap
re-evaluation after each attempt.
"""
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.schemas import QuizCreateRequest, QuizOut, QuestionOut, QuizSubmission, QuizResultOut
from app.models.models import User, StudentProfile, Quiz, Question, QuizAttempt, LearningGap, AuditLog
from app.security.auth import get_current_user
from app.security.rate_limiter import rate_limit
from app.services.quiz_service import generate_quiz_for_topic, evaluate_quiz_submission

router = APIRouter(prefix="/api/quizzes", tags=["Quizzes"])

# Default subject and topic for demo mode (Computer Science / DBMS & SQL JOINs)
_DEFAULT_SUBJECT_ID = 1
_DEFAULT_TOPIC_ID = 2


@router.post("/generate", response_model=QuizOut)
def generate_quiz(
    quiz_req: QuizCreateRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generates a 5-question quiz for the requested topic."""
    rate_limit(request, max_requests=15, window_seconds=60)

    subject_id = quiz_req.subject_id or _DEFAULT_SUBJECT_ID
    topic_id = quiz_req.topic_id or _DEFAULT_TOPIC_ID

    quiz = generate_quiz_for_topic(db, subject_id, topic_id, current_user.id)
    questions = (
        db.query(Question)
        .filter(Question.topic_id == topic_id)
        .limit(5)
        .all()
    )

    audit = AuditLog(
        user_id=current_user.id,
        user_email=current_user.email,
        action="QUIZ_GENERATED",
        details=f"Quiz #{quiz.id} generated for Topic #{topic_id}",
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(audit)
    db.commit()

    q_outs = [
        QuestionOut(
            id=q.id,
            question_text=q.question_text,
            option_a=q.option_a,
            option_b=q.option_b,
            option_c=q.option_c,
            option_d=q.option_d,
            difficulty=q.difficulty
        )
        for q in questions
    ]

    return QuizOut(
        id=quiz.id,
        title=quiz.title,
        subject_id=quiz.subject_id,
        topic_id=quiz.topic_id,
        total_questions=len(q_outs),
        questions=q_outs
    )


@router.post("/{quiz_id}/submit", response_model=QuizResultOut)
def submit_quiz(
    quiz_id: int,
    submission: QuizSubmission,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Evaluates quiz answers and updates the student's learning gap status."""
    sp = current_user.student_profile
    if not sp:
        # Fallback for teacher/admin demo submissions
        sp = db.query(StudentProfile).first()

    attempt = evaluate_quiz_submission(db, quiz_id, sp.id, submission.answers)
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()

    gap = db.query(LearningGap).filter(
        LearningGap.student_id == sp.id,
        LearningGap.topic_id == quiz.topic_id
    ).first()

    audit = AuditLog(
        user_id=current_user.id,
        user_email=current_user.email,
        action="QUIZ_SUBMITTED",
        details=f"Attempt #{attempt.id} for Quiz #{quiz_id}: Score {attempt.score_percentage}%",
        ip_address=request.client.host if request.client else "127.0.0.1"
    )
    db.add(audit)
    db.commit()

    return QuizResultOut(
        attempt_id=attempt.id,
        quiz_id=quiz_id,
        score_percentage=attempt.score_percentage,
        total_questions=attempt.total_questions,
        correct_count=attempt.correct_count,
        incorrect_count=attempt.incorrect_count,
        indicator_status=gap.indicator_status if gap else "ON_TRACK",
        evidence_text=gap.evidence_text if gap else "Assessment completed.",
        completed_at=attempt.completed_at
    )
