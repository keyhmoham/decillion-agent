"""Planning tools: the agent's visible todo list."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .base import ToolError, make_tool
from .context import ToolContext

Status = Literal["pending", "in_progress", "completed"]
MARK = {"pending": " ", "in_progress": "~", "completed": "x"}


class TodoItem(BaseModel):
    content: str = Field(description="One concrete step, imperative mood.")
    status: Status = Field("pending", description="pending | in_progress | completed")


class WriteTodosArgs(BaseModel):
    todos: list[TodoItem] = Field(description="The complete ordered plan (replaces the old one). 3-8 steps.")


class UpdateTodoArgs(BaseModel):
    index: int = Field(description="1-based number of the step.")
    status: Status = Field(description="New status: pending | in_progress | completed")


def render_todos(todos: list[dict[str, Any]]) -> str:
    if not todos:
        return "(no plan yet)"
    done = sum(1 for t in todos if t["status"] == "completed")
    lines = [f"Plan ({done}/{len(todos)} done):"]
    lines += [f"{i}. [{MARK[t['status']]}] {t['content']}" for i, t in enumerate(todos, 1)]
    return "\n".join(lines)


def build_plan_tools(ctx: ToolContext) -> list[BaseTool]:
    async def write_todos(todos: list[Any]) -> str:
        items: list[dict[str, Any]] = []
        for t in todos:
            if isinstance(t, str):
                t = {"content": t, "status": "pending"}
            elif isinstance(t, BaseModel):
                t = t.model_dump()
            content = str(t.get("content", "")).strip()
            if not content:
                continue
            status = t.get("status", "pending")
            items.append({"content": content[:300], "status": status if status in MARK else "pending"})
        if not items:
            raise ToolError("todos is empty. Provide 3-8 short steps.")
        # at most one step in progress
        seen = False
        for it in items:
            if it["status"] == "in_progress":
                if seen:
                    it["status"] = "pending"
                seen = True
        if not seen:
            for it in items:
                if it["status"] == "pending":
                    it["status"] = "in_progress"
                    break
        await ctx.store.set_todos(items)
        await ctx.emit("todos", todos=items)
        return render_todos(items) + "\nNow work on the step marked [~]."

    async def update_todo(index: int, status: str) -> str:
        todos = await ctx.store.get_todos()
        if not todos:
            raise ToolError("There is no plan yet. Call write_todos first.")
        if not 1 <= index <= len(todos):
            raise ToolError(f"index must be between 1 and {len(todos)}.")
        if status not in MARK:
            raise ToolError("status must be pending, in_progress or completed.")
        todos = [dict(t) for t in todos]
        todos[index - 1]["status"] = status
        if status == "completed":  # auto-advance
            if not any(t["status"] == "in_progress" for t in todos):
                for t in todos:
                    if t["status"] == "pending":
                        t["status"] = "in_progress"
                        break
        await ctx.store.set_todos(todos)
        await ctx.emit("todos", todos=todos)
        remaining = [i for i, t in enumerate(todos, 1) if t["status"] != "completed"]
        tail = "All steps are complete - give the final answer." if not remaining else f"Next open step: {remaining[0]}."
        return render_todos(todos) + "\n" + tail

    return [
        make_tool(
            "write_todos",
            "Create or replace your step-by-step plan for a multi-step task. Do this FIRST for any non-trivial task.",
            WriteTodosArgs, write_todos, category="plan",
        ),
        make_tool("update_todo", "Mark one plan step as pending, in_progress or completed.", UpdateTodoArgs, update_todo, category="plan"),
    ]
