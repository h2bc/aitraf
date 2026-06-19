"""Token authentication dependencies."""

from __future__ import annotations

from hmac import compare_digest

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


def require_app_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    settings = request.app.state.settings
    configured_token = settings.api_token
    if not configured_token:
        raise HTTPException(status_code=503, detail="API token is not configured")

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing app token")

    if not compare_digest(credentials.credentials, configured_token):
        raise HTTPException(status_code=401, detail="Invalid app token")


__all__ = ["require_app_token"]
