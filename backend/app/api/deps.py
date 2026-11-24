from __future__ import annotations

import hmac
from typing import Any

from fastapi import Depends, HTTPException, Request, WebSocket, status

from ..agents.service import AgentService
from ..config import Settings


class Container:
    """Everything the API needs, created once in the lifespan."""

    settings: Settings
    db: Any
    repo: Any
    memory: Any
    sandboxes: Any
    mcp: Any
    service: AgentService
    backend: Any


def get_container(request: Request) -> Container:
    return request.app.state.container


def _extract_token(headers, query_token: str | None) -> str:
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return headers.get("x-api-key") or query_token or ""


def check_key(settings: Settings, token: str) -> bool:
    return not settings.app_api_key or hmac.compare_digest(token.encode(), settings.app_api_key.encode())


async def require_auth(request: Request, c: Container = Depends(get_container)) -> None:
    if not check_key(c.settings, _extract_token(request.headers, request.query_params.get("token"))):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key")


def ws_authorized(ws: WebSocket, settings: Settings) -> bool:
    return check_key(settings, _extract_token(ws.headers, ws.query_params.get("token")))
