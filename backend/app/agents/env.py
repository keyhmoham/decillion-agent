"""Shared dependencies of one run, passed to the single-agent graph, workers and the team graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool

from ..config import Settings
from ..db.repo import Repo
from ..llm.chat_model import A1ChatModel
from ..tools.context import ToolContext
from .events import RunRecorder


@dataclass
class RunEnv:
    settings: Settings
    model: A1ChatModel
    recorder: RunRecorder
    ctx: ToolContext
    repo: Repo | None = None
    mcp_tools: list[BaseTool] = field(default_factory=list)
    thinking: bool = True
    memory_enabled: bool = True
    core_memory: dict[str, str] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
