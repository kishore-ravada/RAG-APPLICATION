# EduSaarthi AI

> *"Detect learning gaps before they become learning failures."*

**Category:** AI for Social Impact → AI for Education  
**Architecture:** Modular Monolith (FastAPI + SQLite Backend | React + Vite + Tailwind CSS Frontend)

---

## 📌 Executive Overview

Students rarely become academically weak overnight. Learning difficulties develop gradually through fragmented signals: declining test scores, repeated incorrect answers, topic-level misconceptions, and reduced learning activity. 

**EduSaarthi AI** continuously analyzes these learning signals, detects emerging support needs, explains actionable evidence to educators, generates personalized 5-day interventions, and keeps teachers in full control of all academic decisions.

---

## 🚀 Core Features & Product Loop

1. **DETECT**: Identifies declining performance trajectories (e.g. 82% → 76% → 65% → 52%) and repeated error patterns.
2. **EXPLAIN**: Provides transparent, non-stigmatizing evidence ("Needs Attention") explaining topic accuracy thresholds and error instances.
3. **INTERVENE**: Generates a 5-day personalized practice roadmap for weak concepts.
4. **TEACHER-IN-THE-LOOP**: Teachers APPROVE, EDIT, or REJECT interventions before students receive them.
5. **RAG AI TUTOR**: Interactive educational tutor grounded in reference materials, protected against prompt injection.
6. **DETERMINISTIC QUIZ ENGINE**: Backend calculates exact scores and topic accuracy—never relying on LLMs for calculations.
7. **ADMIN SECURITY HUB**: DevSecOps dashboard displaying real-time security events, rate limiting, and system health status.
8. **DEMO AI MODE**: Seamless fallback to deterministic responses if `GEMINI_API_KEY` is missing or times out.

---

## 🛠️ Technology Stack

- **Backend:** Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy 2.0, SQLite (PostgreSQL ready), JWT Auth, Passlib/PBKDF2, Pytest
- **Frontend:** React 18, Vite, TypeScript, Tailwind CSS, Lucide React, Recharts
- **AI & RAG:** Gemini API, RAG Vector Search with Markdown Knowledge Base (`data/knowledge_base/`)
- **Security:** RAG prompt injection protection, sliding-window rate limiting, HTTP security headers, audit logging

---

## 🔑 Seed Demo Credentials

| Role | Email | Password | Pre-seeded Scenario |
| :--- | :--- | :--- | :--- |
| **Student** | `student@example.com` | `Student123!` | Arjun Sharma — DBMS / SQL JOIN declining score (82% → 76% → 65% → 52%) |
| **Teacher** | `teacher@example.com` | `Teacher123!` | Prof. Sunita Verma — Class roster, evidence review & intervention approval |
| **Admin** | `admin@example.com` | `Admin123!` | System Administrator — Security controls, threat logs & audit trail |

---

## 💻 Quick Start & Installation

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ & npm

### 2. Backend Setup
```bash
cd backend
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe seed.py
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8080
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 🧪 Running Automated Tests

```bash
cd backend
.\venv\Scripts\pytest.exe tests/
```

---

## 🛡️ Responsible AI Commitment
EduSaarthi AI provides AI-assisted learning support indicators and evidence. It does **NOT** diagnose students, label learners as weak or incapable, or make irreversible academic decisions.
