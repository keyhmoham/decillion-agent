"""Catalog of popular MCP servers that run *inside the session sandbox* (stdio via ``docker exec -i``).

The sandbox image pre-installs the packages below, so enabling a server costs a process start,
not a download.  A 4B model degrades with long tool lists, therefore only three small servers are
on by default and the UI shows an approximate token cost per server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class McpPreset:
    id: str
    name: str
    description: str
    argv: tuple[str, ...]
    category: str = "utility"
    default: bool = False
    approx_tools: int = 3
    env_from_settings: dict[str, str] = field(default_factory=dict)  # sandbox env var -> backend env var
    requires: tuple[str, ...] = ()  # backend env vars that must be set
    homepage: str = ""

    def to_dict(self, available: bool = True) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "description": self.description, "category": self.category,
            "default": self.default, "approx_tools": self.approx_tools, "requires": list(self.requires),
            "available": available, "approx_prompt_tokens": self.approx_tools * 110, "homepage": self.homepage,
        }


PRESETS: dict[str, McpPreset] = {
    p.id: p
    for p in [
        McpPreset("time", "Time", "Current time and timezone conversion.", ("mcp-server-time",), "utility", True, 3,
                  homepage="https://github.com/modelcontextprotocol/servers"),
        McpPreset("git", "Git", "Inspect and operate on git repositories (status, diff, log, commit, branch).",
                  ("mcp-server-git", "--repository", "/workspace"), "dev", True, 12),
        McpPreset("sequential-thinking", "Sequential Thinking", "Structured step-by-step problem decomposition tool.",
                  ("mcp-server-sequential-thinking",), "reasoning", True, 1),
        McpPreset("fetch", "Fetch", "Fetch a URL and convert it to markdown (alternative to web_fetch).",
                  ("mcp-server-fetch",), "web", False, 1),
        McpPreset("filesystem", "Filesystem", "Rich file operations (tree, search, move, multi-read). Overlaps native file tools.",
                  ("mcp-server-filesystem", "/workspace"), "dev", False, 12),
        McpPreset("memory", "Knowledge-graph memory", "Entity/relation knowledge graph persisted in the workspace.",
                  ("mcp-server-memory",), "memory", False, 9),
        McpPreset("playwright", "Playwright browser", "Drive a headless browser: navigate, click, screenshot, scrape.",
                  ("playwright-mcp", "--headless", "--no-sandbox"), "web", False, 20),
        McpPreset("context7", "Context7 docs", "Up-to-date library documentation lookup.",
                  ("context7-mcp",), "docs", False, 2),
        McpPreset("sqlite", "SQLite", "Query and modify a SQLite database file.", ("mcp-server-sqlite", "--db-path", "/workspace/data.db"),
                  "data", False, 6),
        McpPreset("github", "GitHub", "Issues, PRs, code search and repository operations (needs a token).",
                  ("mcp-server-github",), "dev", False, 25, env_from_settings={"GITHUB_PERSONAL_ACCESS_TOKEN": "GITHUB_TOKEN"},
                  requires=("GITHUB_TOKEN",)),
        McpPreset("brave-search", "Brave Search", "Web and local search via the Brave API (needs a key).",
                  ("mcp-server-brave-search",), "web", False, 2, env_from_settings={"BRAVE_API_KEY": "BRAVE_API_KEY"},
                  requires=("BRAVE_API_KEY",)),
    ]
}
