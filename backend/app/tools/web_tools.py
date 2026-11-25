"""web_fetch / web_search tools (executed inside the sandbox in docker mode)."""

from __future__ import annotations

import asyncio
import json
import shlex

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from . import web_impl
from .search import SearchError, SearchService, build_providers, format_results
from .base import ToolError, make_tool
from .context import ToolContext

SANDBOX_SCRIPT = "/usr/local/lib/a1_web.py"


class FetchArgs(BaseModel):
    url: str = Field(description="Absolute http(s) URL to read.")
    max_chars: int = Field(8000, description="Maximum characters of page text to return.")


class SearchArgs(BaseModel):
    query: str = Field(description="Web search query. Keep it short and specific.")
    num_results: int = Field(6, description="How many results to return (1-10).")


_DEFAULT: dict[tuple, SearchService] = {}


def get_search_service(ctx: ToolContext) -> SearchService:
    if ctx.search is not None:
        return ctx.search
    s = ctx.settings
    key = (s.search_providers, s.searxng_url, s.brave_api_key)
    if key not in _DEFAULT:
        _DEFAULT[key] = SearchService(build_providers(*key))
    return _DEFAULT[key]


def build_web_tools(ctx: ToolContext) -> list[BaseTool]:
    sb = ctx.sandbox
    cap = ctx.settings.web_fetch_max_chars

    async def run(op: str, arg: str, n: int) -> dict:
        if sb.backend == "docker":
            res = await sb.exec(f"python3 {SANDBOX_SCRIPT} {op} {shlex.quote(arg)} {n}", timeout=60)
            try:
                data = json.loads(res.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError) as exc:
                raise ToolError(f"web helper failed: {(res.stderr or res.stdout)[:300]}") from exc
        else:
            try:
                data = await asyncio.to_thread(web_impl.fetch_url, arg, n)
            except web_impl.WebError as exc:
                raise ToolError(str(exc)) from exc
        if "error" in data:
            raise ToolError(data["error"])
        return data

    async def web_fetch(url: str, max_chars: int = 8000) -> str:
        data = await run("fetch", url, min(max(500, int(max_chars)), cap))
        head = f"{data['url']}" + (f" - {data['title']}" if data.get("title") else "")
        tail = f"\n[truncated: {data['total_chars']} chars total]" if data.get("truncated") else ""
        return f"{head}\n\n{data['text']}{tail}"

    async def web_search(query: str, num_results: int = 6) -> str:
        try:
            results, provider = await get_search_service(ctx).search(query, min(max(1, int(num_results)), 10))
        except SearchError as exc:
            raise ToolError(str(exc)) from exc
        return format_results(results, provider)

    return [
        make_tool("web_search", "Search the web. Returns titles, URLs and snippets. Follow up with web_fetch to read a page.", SearchArgs, web_search, category="web"),
        make_tool("web_fetch", "Download a web page and return it as readable text.", FetchArgs, web_fetch, category="web"),
    ]
