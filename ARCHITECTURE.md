# EduSaarthi AI — Technical Architecture

## Modular Monolith Design

```
edusaarthi-ai/
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI entry point & exception handlers
│   │   ├── config.py              # System configuration & environment loading
│   │   ├── database.py            # SQLAlchemy 2.0 ORM session management
│   │   ├── models/                # SQLAlchemy entities (User, StudentProfile, Quiz, etc.)
│   │   ├── schemas/               # Pydantic v2 schemas for request/response validation
│   │   ├── routers/               # Modular REST endpoints (auth, student, teacher, admin, tutor, quiz)
│   │   ├── services/              # Analytics engine, deterministic scoring, plan generator
│   │   ├── security/              # Auth JWT, RBAC, rate limiter, prompt injection guard
│   │   ├── ai/                    # Gemini API client & Demo Mode fallback
│   │   ├── rag/                   # Document parser, vector search, grounded prompt builder
│   │   └── middleware/            # Security headers & audit loggers
│   └── tests/                     # Pytest suite
├── frontend/                      # React + Vite + TypeScript + Tailwind CSS
├── data/knowledge_base/           # Markdown reference materials for RAG
```

## Data Flow & Closed Feedback Loop

1. **Student Quiz Submission**: Score and individual question answers are submitted to `POST /api/quizzes/{id}/submit`.
2. **Deterministic Analytics**: `app.services.analytics` calculates score percentage, evaluates topic accuracy against thresholds (60%), and checks for 3+ consecutive score drops.
3. **Indicator & Evidence Generation**: System updates `LearningGap` entity with status (`NEEDS_ATTENTION`) and actionable evidence text.
4. **Intervention Generation**: System creates a 5-day `LearningPlan` in `PROPOSED` status.
5. **Teacher Approval**: Teacher inspects evidence on Teacher Dashboard and executes `PATCH /api/teacher/interventions/{id}` (`APPROVE`, `EDIT`, `REJECT`).
6. **Student Execution**: Student follows 5-day roadmap and completes reassessment quiz to record mastery recovery.
