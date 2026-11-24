"""Presentation tools: let the agent render rich, structured cards in the chat UI.

Each tool validates a small, strongly-typed payload (simple enough for a 4B model to fill in
reliably), emits a ``card`` event the frontend renders, and tells the model what the user now sees.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .base import ToolError, make_tool
from .context import ToolContext

MAX_ROWS, MAX_COLS, MAX_POINTS, MAX_SERIES = 200, 20, 300, 12
MAX_HTML, MAX_MERMAID = 60_000, 10_000


class TableArgs(BaseModel):
    title: str = Field(description="Card title.")
    columns: list[str] = Field(description="Column headers, e.g. [\"Name\", \"Score\"].")
    rows: list[list[str | int | float | bool | None]] = Field(description="Table rows; each row has one value per column.")
    caption: str | None = Field(None, description="Optional note under the table.")


class Series(BaseModel):
    name: str = Field(description="Series name.")
    values: list[float] = Field(description="One number per label.")


class ChartArgs(BaseModel):
    title: str = Field(description="Chart title.")
    type: Literal["bar", "line", "area", "pie", "scatter"] = Field(description="Chart type.")
    labels: list[str] = Field(description="X-axis labels (or slice names for pie).")
    series: list[Series] = Field(description="One or more data series; values align with labels.")
    unit: str | None = Field(None, description="Optional unit, e.g. 'ms' or '%'.")


class StatItem(BaseModel):
    label: str = Field(description="Metric name.")
    value: str = Field(description="Metric value as text, e.g. '42' or '99.9%'.")
    delta: str | None = Field(None, description="Optional change, e.g. '+12%'.")
    status: Literal["good", "bad", "neutral"] = Field("neutral", description="Colour hint.")


class StatsArgs(BaseModel):
    title: str = Field(description="Card title.")
    items: list[StatItem] = Field(description="2-8 key metrics.")


class DiagramArgs(BaseModel):
    title: str = Field(description="Diagram title.")
    mermaid: str = Field(description="Mermaid source, e.g. 'graph TD; A-->B'. Flowcharts, sequence, class, ER, gantt are supported.")


class HtmlArgs(BaseModel):
    title: str = Field(description="Card title.")
    html: str = Field(description="Self-contained HTML/SVG/CSS/JS snippet. It runs in an isolated sandbox without network access.")
    height: int = Field(320, description="Card height in pixels (120-900).")


class ImageArgs(BaseModel):
    path: str = Field(description="Path of an image file (png, jpg, gif, svg, webp) in the workspace.")
    caption: str | None = Field(None, description="Optional caption.")


class CalloutArgs(BaseModel):
    title: str = Field(description="Card title.")
    body: str = Field(description="Markdown text.")
    level: Literal["info", "success", "warning", "error"] = Field("info", description="Colour / importance.")
    items: list[str] = Field(default_factory=list, description="Optional bullet list shown under the text.")


def build_card_tools(ctx: ToolContext) -> list[BaseTool]:
    async def emit_card(kind: str, title: str, data: dict[str, Any]) -> str:
        cid = uuid.uuid4().hex[:10]
        await ctx.emit("card", card={"id": cid, "kind": kind, "title": title.strip()[:160], "data": data})
        return f"Card '{title}' ({kind}) is now displayed to the user. Do not repeat its contents in text; add only a short comment if useful."

    async def show_table(title: str, columns: list[str], rows: list[list[Any]], caption: str | None = None) -> str:
        if not columns:
            raise ToolError("columns is empty.")
        if len(columns) > MAX_COLS or len(rows) > MAX_ROWS:
            raise ToolError(f"Table too large (max {MAX_ROWS} rows x {MAX_COLS} columns). Show a summary or the top rows.")
        fixed = []
        for i, r in enumerate(rows):
            if not isinstance(r, list):
                raise ToolError(f"row {i + 1} must be a list of values.")
            r = (list(r) + [None] * len(columns))[: len(columns)]
            fixed.append(r)
        return await emit_card("table", title, {"columns": columns, "rows": fixed, "caption": caption})

    async def show_chart(title: str, type: str, labels: list[str], series: list[Any], unit: str | None = None) -> str:
        norm = []
        for s in series:
            s = s.model_dump() if isinstance(s, BaseModel) else dict(s)
            vals = [float(v) for v in s.get("values", [])]
            norm.append({"name": str(s.get("name", "series")), "values": vals})
        if not labels or not norm:
            raise ToolError("labels and series must not be empty.")
        if len(norm) > MAX_SERIES or len(labels) > MAX_POINTS:
            raise ToolError(f"Chart too large (max {MAX_SERIES} series, {MAX_POINTS} points).")
        for s in norm:
            if len(s["values"]) != len(labels):
                raise ToolError(f"series '{s['name']}' has {len(s['values'])} values but there are {len(labels)} labels. They must match.")
        return await emit_card("chart", title, {"type": type, "labels": labels, "series": norm, "unit": unit})

    async def show_stats(title: str, items: list[Any]) -> str:
        norm = [i.model_dump() if isinstance(i, BaseModel) else dict(i) for i in items]
        if not 1 <= len(norm) <= 12:
            raise ToolError("Provide between 1 and 12 items.")
        return await emit_card("stats", title, {"items": norm})

    async def show_diagram(title: str, mermaid: str) -> str:
        if len(mermaid) > MAX_MERMAID:
            raise ToolError("Diagram source too long.")
        if not mermaid.strip():
            raise ToolError("mermaid is empty.")
        return await emit_card("diagram", title, {"mermaid": mermaid.strip()})

    async def show_html(title: str, html: str, height: int = 320) -> str:
        if len(html) > MAX_HTML:
            raise ToolError(f"HTML too large (max {MAX_HTML} chars).")
        return await emit_card("html", title, {"html": html, "height": max(120, min(int(height), 900))})

    async def show_image(path: str, caption: str | None = None) -> str:
        full = ctx.sandbox.resolve(path)
        if await ctx.sandbox.exists(full) != "file":
            raise ToolError(f"Image not found: {full}")
        if full.rsplit(".", 1)[-1].lower() not in {"png", "jpg", "jpeg", "gif", "svg", "webp"}:
            raise ToolError("Unsupported image type.")
        return await emit_card("image", caption or full.rsplit("/", 1)[-1], {"path": full, "caption": caption})

    async def show_callout(title: str, body: str, level: str = "info", items: list[str] | None = None) -> str:
        return await emit_card("callout", title, {"body": body, "level": level, "items": items or []})

    return [
        make_tool("show_table", "Show a table card to the user. Use for any tabular or list-of-records data.", TableArgs, show_table, category="present"),
        make_tool("show_chart", "Show a chart card (bar, line, area, pie, scatter) to the user. Use for numeric comparisons and trends.", ChartArgs, show_chart, category="present"),
        make_tool("show_stats", "Show a row of KPI/metric tiles to the user.", StatsArgs, show_stats, category="present"),
        make_tool("show_diagram", "Show a Mermaid diagram (flowchart, sequence, ER...) to the user.", DiagramArgs, show_diagram, category="present"),
        make_tool("show_html", "Show a custom visual (HTML/SVG/CSS/JS) in an isolated card, for visualizations no other card can express.", HtmlArgs, show_html, category="present"),
        make_tool("show_image", "Show an image file from the workspace to the user.", ImageArgs, show_image, category="present"),
        make_tool("show_callout", "Show a highlighted summary/notice card with optional bullet points.", CalloutArgs, show_callout, category="present"),
    ]


__all__ = ["build_card_tools", "json"]
