"""
Admin Router — EduSaarthi AI
Restricted endpoints for admin-only security monitoring, audit log access,
and security event inspection.
"""
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.models import User, UserRole, AuditLog, SecurityEvent
from app.schemas.schemas import SecurityDashboardOut, AuditLogOut, SecurityEventOut
from app.security.rbac import require_roles

router = APIRouter(prefix="/api/admin", tags=["Admin Security"])


@router.get("/security-dashboard", response_model=SecurityDashboardOut)
def get_security_dashboard(
    current_user: User = Depends(require_roles([UserRole.ADMIN])),
    db: Session = Depends(get_db)
):
    """
    Returns the full security posture dashboard including:
    active controls, failed login counts, security events, and audit logs.
    """
    total_users = db.query(User).count()
    failed_logins = db.query(AuditLog).filter(AuditLog.action == "LOGIN_FAILED").count()
    unauthorized_attempts = db.query(SecurityEvent).count()

    events = (
        db.query(SecurityEvent)
        .order_by(SecurityEvent.timestamp.desc())
        .limit(20)
        .all()
    )
    logs = (
        db.query(AuditLog)
        .order_by(AuditLog.timestamp.desc())
        .limit(20)
        .all()
    )

    active_controls = {
        "authentication": True,
        "rbac": True,
        "input_validation": True,
        "rate_limiting": True,
        "prompt_injection_protection": True,
        "audit_logging": True,
        "secret_management": True,
        "ai_fallback_demo_mode": True,
        "health_monitoring": True
    }

    return SecurityDashboardOut(
        active_controls=active_controls,
        total_users=total_users,
        failed_logins=failed_logins,
        unauthorized_attempts=unauthorized_attempts,
        security_events=events,
        audit_logs=logs
    )


@router.get("/audit-logs", response_model=List[AuditLogOut])
def get_audit_logs(
    current_user: User = Depends(require_roles([UserRole.ADMIN])),
    db: Session = Depends(get_db)
):
    """Returns the 50 most recent audit log entries."""
    return db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(50).all()


@router.get("/security-events", response_model=List[SecurityEventOut])
def get_security_events(
    current_user: User = Depends(require_roles([UserRole.ADMIN])),
    db: Session = Depends(get_db)
):
    """Returns the 50 most recent security events (e.g., prompt injection attempts)."""
    return db.query(SecurityEvent).order_by(SecurityEvent.timestamp.desc()).limit(50).all()
