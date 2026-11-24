from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    title: str = "New session"
    config: dict[str, Any] = Field(default_factory=dict)


class SessionPatch(BaseModel):
    title: str | None = None
    config: dict[str, Any] | None = None


class SendMessage(BaseModel):
    content: str = Field(min_length=1, max_length=50000)
    options: dict[str, Any] = Field(default_factory=dict)


class ApprovalBody(BaseModel):
    approve: bool
    note: str = ""


class FileWrite(BaseModel):
    path: str
    content: str


class TerminalExec(BaseModel):
    command: str = Field(min_length=1, max_length=20000)
    timeout: int = 60
    cwd: str | None = None


class MemoryCreate(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    kind: Literal["fact", "preference", "episode", "lesson", "project"] = "fact"
    importance: int = 3
    tags: list[str] = Field(default_factory=list)
    scope: Literal["global", "session"] = "global"
    session_id: str | None = None


class MemorySearch(BaseModel):
    query: str
    session_id: str | None = None
    limit: int = 8


class CoreMemoryBody(BaseModel):
    content: str = Field(max_length=1200)
