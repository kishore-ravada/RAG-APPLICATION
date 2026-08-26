"""
Analytics Service — EduSaarthi AI
Deterministic engine that evaluates a student's learning gap status
for a specific topic based on quiz attempt history.
"""
from typing import Tuple
from sqlalchemy.orm import Session

from app.models.models import QuizAttempt, LearningGap, IndicatorStatus, Topic, Quiz


def evaluate_student_learning_gaps(
    db: Session,
    student_id: int,
    topic_id: int
) -> Tuple[IndicatorStatus, str, float, int]:
    """
    Calculates a student's learning gap status for a specific topic.

    Evaluates:
    - Most recent quiz accuracy for the topic
    - Whether scores show a consecutive declining trajectory
    - Number of incorrect answers accumulated on this topic

    Returns:
        Tuple of (IndicatorStatus, evidence_text, accuracy_percentage, repeated_mistakes_count)
    """
    # Only fetch attempts for quizzes belonging to the specific topic
    attempts = (
        db.query(QuizAttempt)
        .join(Quiz, Quiz.id == QuizAttempt.quiz_id)
        .filter(
            QuizAttempt.student_id == student_id,
            Quiz.topic_id == topic_id
        )
        .order_by(QuizAttempt.completed_at.asc())
        .all()
    )

    if not attempts:
        return IndicatorStatus.ON_TRACK, "Insufficient assessment data recorded yet.", 100.0, 0

    scores = [a.score_percentage for a in attempts]
    recent_accuracy = scores[-1]

    # Detect three consecutive declining scores
    is_declining = len(scores) >= 3 and scores[-1] < scores[-2] < scores[-3]

    # Count incorrect answers scoped to this topic only
    repeated_mistakes = sum(a.incorrect_count for a in attempts)

    topic = db.query(Topic).filter(Topic.id == topic_id).first()
    topic_name = topic.name if topic else "Selected Topic"

    evidence_items = []

    if recent_accuracy < 60.0 or is_declining:
        status = IndicatorStatus.NEEDS_ATTENTION
        if recent_accuracy < 60.0:
            evidence_items.append(
                f"Topic accuracy ({recent_accuracy:.0f}%) is below the 60% mastery threshold."
            )
        if is_declining:
            trend_str = " → ".join([f"{s:.0f}%" for s in scores[-4:]])
            evidence_items.append(
                f"Assessment trajectory shows consecutive decline ({trend_str})."
            )
        if repeated_mistakes > 2:
            evidence_items.append(
                f"Repeated incorrect responses detected ({repeated_mistakes} error instances in {topic_name})."
            )
    elif recent_accuracy >= 80.0:
        status = IndicatorStatus.STRONG
        evidence_items.append(
            f"Consistently high accuracy ({recent_accuracy:.0f}%) demonstrating strong topic mastery."
        )
    else:
        status = IndicatorStatus.ON_TRACK
        evidence_items.append(
            f"Steady performance ({recent_accuracy:.0f}%) meeting expected learning milestones."
        )

    evidence_text = " | ".join(evidence_items)
    return status, evidence_text, recent_accuracy, repeated_mistakes
