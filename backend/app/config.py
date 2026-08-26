"""
Application Configuration — EduSaarthi AI
Centralised settings loaded from environment variables with safe defaults
for local development. Set GEMINI_API_KEY in your .env to enable live AI.
"""
import os


class Settings:
    PROJECT_NAME: str = "EduSaarthi AI"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    # Database — defaults to local SQLite for development
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./edusaarthi.db")

    # JWT Authentication
    SECRET_KEY: str = os.getenv(
        "JWT_SECRET_KEY",
        "edusaarthi-super-secret-jwt-key-2026-hackathon-security"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day

    # Google Gemini AI — leave blank to run in deterministic demo mode
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Rate limiting (requests per minute, applied per IP)
    RATE_LIMIT_PER_MINUTE: int = 30

    # CORS — allow Vite dev server and standard localhost ports
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]


settings = Settings()
