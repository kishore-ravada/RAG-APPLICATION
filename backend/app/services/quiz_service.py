from typing import List, Dict
from sqlalchemy.orm import Session
from app.models.models import Quiz, Question, QuizAttempt, QuizAnswer, LearningGap, Topic
from app.schemas.schemas import AnswerInput
from app.services.analytics import evaluate_student_learning_gaps

def generate_quiz_for_topic(db: Session, subject_id: int, topic_id: int, user_id: int) -> Quiz:
    """Fetches questions for given topic or creates a standard 5-question quiz."""
    topic = db.query(Topic).filter(Topic.id == topic_id).first()
    title = f"{topic.name if topic else 'Assessment'} Quiz"
    
    quiz = Quiz(
        title=title,
        subject_id=subject_id,
        topic_id=topic_id,
        total_questions=5,
        created_by_id=user_id
    )
    db.add(quiz)
    db.commit()
    db.refresh(quiz)
    return quiz

def evaluate_quiz_submission(db: Session, quiz_id: int, student_profile_id: int, user_answers: List[AnswerInput]) -> QuizAttempt:
    """
    Deterministic quiz evaluation:
    Calculates correct/incorrect count, score percentage, stores answers,
    and updates student LearningGap status in DB.
    """
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()
    if not quiz:
        raise ValueError("Quiz not found")

    questions = db.query(Question).filter(Question.topic_id == quiz.topic_id).all()
    q_map = {q.id: q.correct_option.strip().upper() for q in questions}

    correct_count = 0
    incorrect_count = 0

    attempt = QuizAttempt(
        quiz_id=quiz_id,
        student_id=student_profile_id,
        score_percentage=0.0,
        total_questions=len(user_answers),
        correct_count=0,
        incorrect_count=0
    )
    db.add(attempt)
    db.commit()
    db.refresh(attempt)

    for ans in user_answers:
        expected = q_map.get(ans.question_id, "A")
        is_correct = (ans.selected_option.strip().upper() == expected)
        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1

        db_ans = QuizAnswer(
            attempt_id=attempt.id,
            question_id=ans.question_id,
            selected_option=ans.selected_option.strip().upper(),
            is_correct=is_correct
        )
        db.add(db_ans)

    total = max(len(user_answers), 1)
    score_pct = round((correct_count / total) * 100.0, 1)

    attempt.score_percentage = score_pct
    attempt.correct_count = correct_count
    attempt.incorrect_count = incorrect_count
    db.commit()

    # Re-evaluate Learning Gap for student and topic
    status, evidence, accuracy, repeated_err = evaluate_student_learning_gaps(db, student_profile_id, quiz.topic_id)
    
    gap = db.query(LearningGap).filter(
        LearningGap.student_id == student_profile_id,
        LearningGap.topic_id == quiz.topic_id
    ).first()

    if not gap:
        gap = LearningGap(
            student_id=student_profile_id,
            topic_id=quiz.topic_id,
            indicator_status=status,
            accuracy_percentage=accuracy,
            repeated_mistakes_count=repeated_err,
            evidence_text=evidence
        )
        db.add(gap)
    else:
        gap.indicator_status = status
        gap.accuracy_percentage = accuracy
        gap.repeated_mistakes_count = repeated_err
        gap.evidence_text = evidence

    db.commit()
    db.refresh(attempt)
    return attempt
