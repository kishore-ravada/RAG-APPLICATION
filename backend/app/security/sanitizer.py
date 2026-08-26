import re
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from app.models.models import SecurityEvent

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"reveal\s+(your\s+)?system\s+prompt",
    r"show\s+(me\s+)?the\s+gemini\s+api\s+key",
    r"tell\s+me\s+your\s+hidden\s+instructions",
    r"bypass\s+security",
    r"drop\s+table",
    r"<script>",
    r"export\s+env",
]

def sanitize_input(text: str) -> str:
    """Strip basic HTML tags and whitespace."""
    if not text:
        return ""
    clean = re.sub(r'<[^>]*>', '', text)
    return clean.strip()

def check_prompt_injection(text: str, client_ip: str = "127.0.0.1", db: Session = None) -> bool:
    """
    Scans input for prompt injection attack patterns.
    Logs security event and raises 400 Bad Request if detected.
    """
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            if db:
                event = SecurityEvent(
                    event_type="PROMPT_INJECTION_ATTEMPT",
                    severity="HIGH",
                    description=f"Attempted prompt injection payload: '{text[:80]}...'",
                    source_ip=client_ip
                )
                db.add(event)
                db.commit()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Security policy violation: Prompt injection or restricted instruction detected."
            )
    return True
