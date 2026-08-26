from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from datetime import datetime
from app.models.models import UserRole, IndicatorStatus, PlanStatus

# --- Auth Schemas ---
class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str
    role: UserRole = UserRole.STUDENT

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    email: str
    full_name: str
    user_id: int

class UserOut(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    role: UserRole
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

# --- Quiz Schemas ---
class QuestionOut(BaseModel):
    id: int
    question_text: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    difficulty: str

    class Config:
        from_attributes = True

class QuizCreateRequest(BaseModel):
    subject_id: Optional[int] = 1
    topic_id: Optional[int] = 1
    difficulty: Optional[str] = "INTERMEDIATE"

class QuizOut(BaseModel):
    id: int
    title: str
    subject_id: int
    topic_id: int
    total_questions: int
    questions: List[QuestionOut]

class AnswerInput(BaseModel):
    question_id: int
    selected_option: str  # A, B, C, D

class QuizSubmission(BaseModel):
    answers: List[AnswerInput]

class QuizResultOut(BaseModel):
    attempt_id: int
    quiz_id: int
    score_percentage: float
    total_questions: int
    correct_count: int
    incorrect_count: int
    indicator_status: IndicatorStatus
    evidence_text: str
    completed_at: datetime

# --- AI Tutor Schemas ---
class TutorRequest(BaseModel):
    question: str = Field(..., max_length=1000)
    level: str = "INTERMEDIATE"  # BEGINNER, INTERMEDIATE, ADVANCED

class SourceOut(BaseModel):
    title: str
    snippet: str

class TutorResponse(BaseModel):
    explanation: str
    example: str
    common_mistake: str
    practice_question: str
    sources: List[SourceOut]
    is_demo_mode: bool = False

# --- Learning Gap & Plan Schemas ---
class GapOut(BaseModel):
    id: int
    topic_name: str
    indicator_status: IndicatorStatus
    accuracy_percentage: float
    repeated_mistakes_count: int
    evidence_text: str
    updated_at: datetime

class PlanItemOut(BaseModel):
    id: int
    day_number: int
    title: str
    objective: str
    activity_type: str
    resource_link: Optional[str] = None
    status: str

    class Config:
        from_attributes = True

class PlanOut(BaseModel):
    id: int
    student_id: int
    topic_id: int
    topic_name: str
    title: str
    status: PlanStatus
    created_by: str
    teacher_feedback: Optional[str] = None
    created_at: datetime
    items: List[PlanItemOut]

    class Config:
        from_attributes = True

class PlanActionInput(BaseModel):
    action: str  # APPROVE, REJECT, EDIT
    teacher_feedback: Optional[str] = None

# --- Dashboard Schemas ---
class TrendPoint(BaseModel):
    attempt_number: int
    score: float
    date: str

class StudentDashOut(BaseModel):
    student_name: str
    grade_level: str
    indicator_status: IndicatorStatus
    average_score: float
    strong_topics: List[str]
    weak_topics: List[str]
    recent_trend: List[TrendPoint]
    active_plan: Optional[PlanOut] = None

class StudentOverviewForTeacher(BaseModel):
    student_id: int
    user_id: int
    full_name: str
    grade_level: str
    average_score: float
    indicator_status: IndicatorStatus
    weakest_topic: str
    evidence_snippet: str
    recommended_action: str

class TeacherDashOut(BaseModel):
    total_students: int
    students_on_track: int
    students_needing_attention: int
    average_class_score: float
    most_difficult_topics: List[str]
    students: List[StudentOverviewForTeacher]

class StudentDetailOut(BaseModel):
    student_id: int
    full_name: str
    grade_level: str
    indicator_status: IndicatorStatus
    average_score: float
    score_history: List[TrendPoint]
    learning_gaps: List[GapOut]
    active_plans: List[PlanOut]

class AuditLogOut(BaseModel):
    id: int
    user_email: Optional[str]
    action: str
    details: Optional[str]
    ip_address: Optional[str]
    timestamp: datetime

    class Config:
        from_attributes = True

class SecurityEventOut(BaseModel):
    id: int
    event_type: str
    severity: str
    description: str
    source_ip: Optional[str]
    timestamp: datetime

    class Config:
        from_attributes = True

class SecurityDashboardOut(BaseModel):
    active_controls: dict
    total_users: int
    failed_logins: int
    unauthorized_attempts: int
    security_events: List[SecurityEventOut]
    audit_logs: List[AuditLogOut]

class HealthCheckOut(BaseModel):
    status: str
    database: str
    ai: str
    rag: str
    timestamp: datetime
