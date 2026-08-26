from typing import List
from fastapi import HTTPException, status, Depends
from app.models.models import User, UserRole, StudentProfile
from app.security.auth import get_current_user

def require_roles(allowed_roles: List[UserRole]):
    def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {[r.value for r in allowed_roles]}"
            )
        return current_user
    return role_checker

def check_student_ownership(student_profile_id: int, current_user: User):
    """
    Ensure students can only access their own student profile data.
    Teachers and Admins bypass student ownership checks for assigned view.
    """
    if current_user.role == UserRole.STUDENT:
        if not current_user.student_profile or current_user.student_profile.id != student_profile_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. You can only view your own student data."
            )
