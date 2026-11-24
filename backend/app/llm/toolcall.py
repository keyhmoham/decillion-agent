"""Parsing of the model's native tool-call syntax.

Agents-A1-4B emits (Qwen3-Coder / Qwen3.5 XML convention)::

    <tool_call>
    <function=write_file>
    <parameter=path>
    /workspace/a.py
    </parameter>
    <parameter=content>
    print(1)
    </parameter>
    </function>
    </tool_call>

All parameter values are *text*; types must be recovered from the tool's JSON schema.  Because
this is a 4B model we are deliberately tolerant: we also accept Hermes-style JSON bodies, missing
closing tags (truncated generations), quoted function names and a few other near misses.
"""

from __future__ import annotations

import ast
import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Any

import jsonschema

_FUNC_RE = re.compile(r"<function\s*=\s*[\"']?([^>\"'\s]+)[\"']?\s*>", re.IGNORECASE)
_PARAM_RE = re.compile(
    r"<parameter\s*=\s*[\"']?([^>\"'\s]+)[\"']?\s*>(.*?)(?:</parameter>|(?=<parameter\s*=)|(?=</function>)|\Z)",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    error: str | None = None  # populated for calls we could not make sense of
    repaired: bool = False  # true when we had to be lenient (JSON body, truncated tags, ...)


def _strip_one_newline(value: str) -> str:
    if value.startswith("\r\n"):
        value = value[2:]
    elif value.startswith("\n"):
        value = value[1:]
    if value.endswith("\r\n"):
        value = value[:-2]
    elif value.endswith("\n"):
        value = value[:-1]
    return value


def parse_tool_call_body(body: str) -> ParsedToolCall:
    """Parse the text found between ``<tool_call>`` and ``</tool_call>``."""
    raw = body
    text = body.strip()
    if not text:
        return ParsedToolCall(name="", raw=raw, error="Empty <tool_call> block.")

    # --- Native format -------------------------------------------------------------------
    m = _FUNC_RE.search(text)
    if m:
        name = m.group(1).strip()
        rest = text[m.end():]
        end = rest.rfind("</function>")
        inner = rest if end == -1 else rest[:end]
        repaired = end == -1
        args: dict[str, Any] = {}
        for pm in _PARAM_RE.finditer(inner):
            key = pm.group(1).strip()
            args[key] = _strip_one_newline(pm.group(2))
            if not pm.group(0).rstrip().endswith("</parameter>"):
                repaired = True
        if not args:
            # <function=x>{"a": 1}</function>  -> JSON arguments inside the function tag
            stripped = inner.strip()
            if stripped.startswith("{"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict):
                        return ParsedToolCall(name, parsed, raw, repaired=True)
                except json.JSONDecodeError:
                    pass
        return ParsedToolCall(name, args, raw, repaired=repaired)

    # --- Hermes-style JSON ---------------------------------------------------------------
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            name = obj.get("name") or (obj.get("function") or {}).get("name")
            arguments = obj.get("arguments", obj.get("parameters", obj.get("args", {})))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if name and isinstance(arguments, dict):
                return ParsedToolCall(str(name), arguments, raw, repaired=True)

    return ParsedToolCall(
        name="", raw=raw,
        error="Could not find '<function=NAME>' inside the <tool_call> block.",
    )


# --------------------------------------------------------------------------------------------
# Schema-guided coercion
# --------------------------------------------------------------------------------------------

def _schema_types(prop: dict[str, Any]) -> set[str]:
    types: set[str] = set()
    t = prop.get("type")
    if isinstance(t, str):
        types.add(t)
    elif isinstance(t, list):
        types.update(x for x in t if isinstance(x, str))
    for key in ("anyOf", "oneOf"):
        for sub in prop.get(key, []) or []:
            if isinstance(sub, dict):
                types |= _schema_types(sub)
    if "enum" in prop and not types:
        types.add("string")
    return types


def _coerce_value(value: Any, prop: dict[str, Any]) -> Any:
    if not isinstance(value, str):
        return value
    types = _schema_types(prop)
    if not types or types == {"string"}:
        return value
    stripped = value.strip()

    if "null" in types and stripped.lower() in {"null", "none"}:
        return None
    if "boolean" in types and stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    if "integer" in types:
        try:
            return int(stripped)
        except ValueError:
            try:
                f = float(stripped)
                if f.is_integer():
                    return int(f)
            except ValueError:
                pass
    if "number" in types:
        try:
            return float(stripped) if not re.fullmatch(r"[+-]?\d+", stripped) else int(stripped)
        except ValueError:
            pass
    if types & {"array", "object"}:
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(stripped)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(parsed, (list, tuple)) and "array" in types:
                return [_coerce_item(x, prop.get("items")) for x in parsed]
            if isinstance(parsed, dict) and "object" in types:
                return parsed
    if "string" in types:
        return value
    return value


def _coerce_item(item: Any, schema: dict[str, Any] | None) -> Any:
    if not isinstance(schema, dict):
        return item
    if isinstance(item, dict) and schema.get("type") == "object":
        props = schema.get("properties", {})
        return {k: _coerce_value(v, props.get(k, {})) if isinstance(v, str) else v for k, v in item.items()}
    return _coerce_value(item, schema) if isinstance(item, str) else item


def coerce_arguments(args: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any]:
    """Convert the all-string XML parameter values to the types declared by ``schema``."""
    if not schema:
        return args
    props = schema.get("properties", {}) or {}
    out: dict[str, Any] = {}
    for key, value in args.items():
        prop = props.get(key)
        out[key] = _coerce_value(value, prop) if isinstance(prop, dict) else value
    # Items inside arrays/objects that arrived already parsed
    for key, prop in props.items():
        if key in out and isinstance(out[key], list) and isinstance(prop, dict):
            out[key] = [_coerce_item(x, prop.get("items")) for x in out[key]]
    return out


def validate_arguments(args: dict[str, Any], schema: dict[str, Any] | None) -> list[str]:
    """Return human-readable validation problems (empty list when valid)."""
    if not schema:
        return []
    validator = jsonschema.Draft202012Validator(schema)
    problems: list[str] = []
    for err in sorted(validator.iter_errors(args), key=lambda e: list(e.absolute_path)):
        loc = ".".join(str(p) for p in err.absolute_path) or "(arguments)"
        problems.append(f"{loc}: {err.message}")
    return problems[:6]


def resolve_tool_name(name: str, available: list[str]) -> str | None:
    """Map a (possibly slightly wrong) tool name onto an available one."""
    if name in available:
        return name
    lowered = {a.lower(): a for a in available}
    if name.lower() in lowered:
        return lowered[name.lower()]
    norm = re.sub(r"[^a-z0-9]", "", name.lower())
    for a in available:
        if re.sub(r"[^a-z0-9]", "", a.lower()) == norm:
            return a
    close = difflib.get_close_matches(name, available, n=1, cutoff=0.82)
    return close[0] if close else None


def format_call_for_display(name: str, args: dict[str, Any], limit: int = 200) -> str:
    parts = []
    for k, v in args.items():
        s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        s = s.replace("\n", "\\n")
        parts.append(f"{k}={s[:limit] + '…' if len(s) > limit else s}")
    return f"{name}({', '.join(parts)})"


def compact_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Strip noise (titles, additionalProperties) from an OpenAI tool schema - every token counts
    for a 4B model and the system prompt carries the full list on every request.

    Careful: a *property* may itself be called ``title``; only schema metadata is removed.
    """

    def clean(node: Any, in_properties: bool = False) -> Any:
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if not in_properties and (k in {"title", "additionalProperties"} or (k == "default" and v is None)):
                    continue
                out[k] = clean(v, in_properties=(k == "properties" and not in_properties))
            return out
        if isinstance(node, list):
            return [clean(x) for x in node]
        return node

    return clean(tool)
