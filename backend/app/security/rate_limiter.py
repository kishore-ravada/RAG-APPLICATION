import time
from collections import defaultdict
from fastapi import Request, HTTPException, status
from app.config import settings

# Simple in-memory sliding window rate limiter
_request_timestamps = defaultdict(list)

def rate_limit(request: Request, max_requests: int = settings.RATE_LIMIT_PER_MINUTE, window_seconds: int = 60):
    client_ip = request.client.host if request.client else "127.0.0.1"
    now = time.time()
    
    # Filter out timestamps outside the sliding window
    timestamps = [t for t in _request_timestamps[client_ip] if now - t < window_seconds]
    _request_timestamps[client_ip] = timestamps

    if len(timestamps) >= max_requests:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again in a minute."
        )

    _request_timestamps[client_ip].append(now)
