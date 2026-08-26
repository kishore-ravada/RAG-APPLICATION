from sqlalchemy.orm import Session
from app.models.models import LearningPlan, LearningPlanItem, Topic, PlanStatus
from app.ai.gemini_client import generate_ai_learning_plan

def create_or_get_learning_plan(db: Session, student_profile_id: int, topic_id: int) -> LearningPlan:
    """Generates or retrieves a 5-day personalized intervention plan for student."""
    existing = db.query(LearningPlan).filter(
        LearningPlan.student_id == student_profile_id,
        LearningPlan.topic_id == topic_id,
        LearningPlan.status != PlanStatus.REJECTED
    ).first()

    if existing:
        return existing

    topic = db.query(Topic).filter(Topic.id == topic_id).first()
    topic_name = topic.name if topic else "Target Subject"

    plan = LearningPlan(
        student_id=student_profile_id,
        topic_id=topic_id,
        title=f"5-Day Learning Support Plan: {topic_name}",
        status=PlanStatus.PROPOSED,
        created_by="SYSTEM"
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)

    items_data = generate_ai_learning_plan(topic_name)
    for item in items_data:
        db_item = LearningPlanItem(
            plan_id=plan.id,
            day_number=item["day_number"],
            title=item["title"],
            objective=item["objective"],
            activity_type=item.get("activity_type", "Concept Review"),
            resource_link=item.get("resource_link", "#")
        )
        db.add(db_item)

    db.commit()
    db.refresh(plan)
    return plan
