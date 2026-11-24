"""Agent roles for team mode.

Roles are deliberately few and sharply different: a 4B model follows a narrow brief far more
reliably than a broad one, and a small tool list (fewer tokens, fewer wrong choices) matters more
than extra capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Role:
    name: str
    title: str
    description: str          # shown to the orchestrator when it plans
    brief: str                # injected into the worker's system prompt
    tool_groups: tuple[str, ...] = ("sandbox",)
    deny_tools: tuple[str, ...] = ()
    thinking: bool = True
    thinking_budget: int | None = None
    temperature: float | None = None
    max_steps: int = 24
    read_only: bool = False
    color: str = "#6366f1"


ROLES: dict[str, Role] = {
    r.name: r
    for r in [
        Role(
            "researcher", "Researcher",
            "Finds information on the web and in files, verifies facts, writes findings to notes files.",
            "You research. Search the web, open the best sources, cross-check facts, and record concise findings "
            "(with URLs) in a notes file in the workspace when the result is large. Never invent sources.",
            tool_groups=("web", "readonly", "present_min", "memory_read"), thinking_budget=1500, color="#0ea5e9", max_steps=18,
        ),
        Role(
            "coder", "Coder",
            "Writes and modifies code and files, runs it, and fixes errors until it works.",
            "You implement. Write clean, working code. Run it to verify. Fix errors you see. Keep changes minimal and focused "
            "on your task; other agents own the rest.",
            tool_groups=("sandbox", "plan", "memory_read"), thinking_budget=1400, color="#22c55e", max_steps=28,
        ),
        Role(
            "tester", "Tester",
            "Writes and runs tests, reproduces bugs, and reports exact failures.",
            "You verify. Write focused tests (or run existing ones), execute them, and report precisely what passed and what failed "
            "with the failing output. Fix test code freely; for product bugs report them with the file, line and a suggested fix.",
            tool_groups=("sandbox", "memory_read"), thinking_budget=1200, color="#f59e0b", max_steps=22,
        ),
        Role(
            "reviewer", "Reviewer",
            "Reads code and results critically; finds bugs, risks and gaps; does NOT modify files.",
            "You review. Read the relevant files, run read-only checks (linters, type checks, tests) and list concrete problems "
            "ordered by severity, each with file:line and a fix. If everything is fine say so plainly. Do not modify files.",
            tool_groups=("readonly", "bash_ro"), read_only=True, thinking_budget=1500, temperature=0.4, color="#ef4444", max_steps=16,
        ),
        Role(
            "writer", "Writer",
            "Produces documentation, READMEs, reports and other prose artifacts.",
            "You write. Produce clear, accurate documents grounded in the actual files and results in the workspace. "
            "Read before you write; never describe features that do not exist.",
            tool_groups=("sandbox", "web_read", "present", "memory_read"), thinking=False, color="#a855f7", max_steps=16,
        ),
        Role(
            "devops", "DevOps",
            "Sets up environments, installs dependencies, builds, packages and runs services.",
            "You prepare and operate environments: install dependencies, configure, build, start services in the background and "
            "check they respond. Prefer non-interactive commands. Report the exact commands and ports.",
            tool_groups=("sandbox",), thinking_budget=1200, color="#14b8a6", max_steps=22,
        ),
    ]
}
ROLE_NAMES: tuple[str, ...] = tuple(ROLES)

GENERALIST = Role(
    "generalist", "Agent", "General purpose agent.",
    "You are a capable general-purpose engineering agent.",
    tool_groups=("sandbox", "web", "plan", "present", "memory", "team"), color="#6366f1", max_steps=40,
)


def get_role(name: str) -> Role:
    return ROLES.get(name) or GENERALIST


def roles_for_prompt() -> str:
    return "\n".join(f"- {r.name}: {r.description}" for r in ROLES.values())


__all__ = ["Role", "ROLES", "ROLE_NAMES", "GENERALIST", "get_role", "roles_for_prompt", "field"]
